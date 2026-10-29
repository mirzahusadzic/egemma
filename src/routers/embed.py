import logging
import math
from enum import Enum
from functools import lru_cache
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import settings
from ..models.embedding import SentenceTransformerWrapper
from ..util import is_likely_binary, read_upload
from ..util.rate_limiter import get_in_memory_rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingDimensions(int, Enum):
    DIM_128 = 128
    DIM_256 = 256
    DIM_512 = 512
    DIM_768 = 768


class PromptNames(str, Enum):
    QUERY = "query"
    DOCUMENT = "document"
    BITEXT_MINING = "BitextMining"
    CLUSTERING = "Clustering"
    CLASSIFICATION = "Classification"
    INSTRUCTION_RETRIEVAL = "InstructionRetrieval"
    MULTILABEL_CLASSIFICATION = "MultilabelClassification"
    PAIR_CLASSIFICATION = "PairClassification"
    RERANKING = "Reranking"
    RETRIEVAL = "Retrieval"
    RETRIEVAL_QUERY = "Retrieval-query"
    RETRIEVAL_DOCUMENT = "Retrieval-document"
    STS = "STS"
    SUMMARIZATION = "Summarization"


class TextEmbeddingBatchRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        max_length=settings.EMBEDDING_BATCH_MAX_INPUTS,
    )
    dimensions: EmbeddingDimensions = EmbeddingDimensions.DIM_128
    prompt_name: PromptNames = PromptNames.STS


class TextEmbeddingBatchResponse(BaseModel):
    model: str
    revision: str
    dimensions: int
    prompt_name: str
    embeddings: list[list[float]]


class TokenSpanItem(BaseModel):
    text: str
    spans: list[tuple[int, int]] = Field(min_length=1, max_length=64)


class TokenStatesRequest(BaseModel):
    items: list[TokenSpanItem] = Field(
        min_length=1,
        max_length=settings.EMBEDDING_BATCH_MAX_INPUTS,
    )


class TokenStatesItem(BaseModel):
    vectors: list[list[float] | None]


class TokenStatesResponse(BaseModel):
    model: str
    revision: str
    layer: str
    dimensions: int
    items: list[TokenStatesItem]


# Shared model wrapper instance (initialized in server.py)
embedding_model_wrapper: SentenceTransformerWrapper | None = None


@lru_cache(maxsize=settings.CACHE_MAX_SIZE)
def cached_encode(text: str, prompt_name: str | None = None, title: str | None = None):
    return embedding_model_wrapper.encode(text, prompt_name=prompt_name, title=title)


DEFAULT_DIMENSIONS_QUERY = Query(
    [EmbeddingDimensions.DIM_128], description="Select embedding dimensions"
)

DEFAULT_PROMPT_NAME_QUERY = Query(
    PromptNames.DOCUMENT, description="Select a prompt for the embedding model"
)

# Create router
router = APIRouter(tags=["Embeddings"])


@router.post(
    "/embed/texts",
    summary="Embed a bounded text batch",
    response_model=TextEmbeddingBatchResponse,
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.EMBED_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.EMBED_RATE_LIMIT_CALLS,
            )
        )
    ],
)
async def embed_texts(request: TextEmbeddingBatchRequest):
    if embedding_model_wrapper is None or embedding_model_wrapper.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not loaded.",
        )
    if any(
        not text or len(text) > settings.EMBEDDING_MAX_INPUT_CHARS
        for text in request.texts
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Every text must be nonempty and within the configured character bound."
            ),
        )

    try:
        encoded = await run_in_threadpool(
            embedding_model_wrapper.encode_many,
            request.texts,
            request.prompt_name,
        )
        dimensions = request.dimensions.value
        if len(encoded) != len(request.texts):
            raise RuntimeError("Embedding result count does not match input count")
        embeddings = [row[:dimensions].tolist() for row in encoded]
        if any(
            len(vector) != dimensions
            or any(not math.isfinite(coordinate) for coordinate in vector)
            for vector in embeddings
        ):
            raise RuntimeError("Embedding result has invalid coordinates")
        return TextEmbeddingBatchResponse(
            model=settings.EMBEDDING_MODEL_NAME,
            revision=settings.EMBEDDING_MODEL_REVISION,
            dimensions=dimensions,
            prompt_name=request.prompt_name.value,
            embeddings=embeddings,
        )
    except Exception as exc:
        logger.exception("Batch embedding failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding failed: {exc}",
        ) from exc


@router.post(
    "/embed",
    summary="Embed File Content",
    description=(
        "Embeds the content of an uploaded file using the Gemma embedding 300m model "
        "with Matryoshka support."
    ),
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.EMBED_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.EMBED_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def embed(
    file: Annotated[UploadFile, File(...)],
    dimensions: List[EmbeddingDimensions] | None = DEFAULT_DIMENSIONS_QUERY,
    prompt_name: PromptNames | None = DEFAULT_PROMPT_NAME_QUERY,
    title: str | None = Query(
        None, description="Optional title for the embedded content."
    ),
):
    if dimensions is None:
        dimensions = [EmbeddingDimensions.DIM_128]
    try:
        file_content_bytes = await read_upload(file)
        if is_likely_binary(file_content_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file appears to be binary and cannot be embedded.",
            )
        file_content = file_content_bytes.decode("utf-8")

        embedding = await run_in_threadpool(
            cached_encode, file_content, prompt_name, title
        )

        if not dimensions:
            return {"embedding": embedding.tolist()}

        result = {}
        for dim in dimensions:
            dim_value = dim.value
            if dim_value in [128, 256, 512, 768]:
                result[f"embedding_{dim_value}d"] = embedding[:dim_value].tolist()

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        ) from e


@router.post(
    "/embed/tokens",
    summary="The last layer's state of character spans in their text",
    response_model=TokenStatesResponse,
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.EMBED_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.EMBED_RATE_LIMIT_CALLS,
            )
        )
    ],
)
async def embed_tokens(request: TokenStatesRequest):
    """A word read in its sentence: for each span, the mean of the last layer's
    states of the tokens it overlaps (the text read as written, no prompt; the
    full width, unnormalized)."""
    if embedding_model_wrapper is None or embedding_model_wrapper.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not loaded.",
        )
    for item in request.items:
        if not item.text or len(item.text) > settings.EMBEDDING_MAX_INPUT_CHARS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Every text must be nonempty and within the configured "
                    "character bound."
                ),
            )
        if any(not (0 <= a < b <= len(item.text)) for a, b in item.spans):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Every span must lie within its text and be nonempty.",
            )
    try:
        states = await run_in_threadpool(
            embedding_model_wrapper.token_states,
            [
                (item.text, [tuple(span) for span in item.spans])
                for item in request.items
            ],
        )
        items, width = [], None
        for row in states:
            vectors = []
            for vector in row:
                if vector is None:
                    vectors.append(None)
                    continue
                values = vector.tolist()
                if any(not math.isfinite(v) for v in values):
                    raise RuntimeError("Token state has invalid coordinates")
                width = width or len(values)
                vectors.append(values)
            items.append(TokenStatesItem(vectors=vectors))
        return TokenStatesResponse(
            model=settings.EMBEDDING_MODEL_NAME,
            revision=settings.EMBEDDING_MODEL_REVISION,
            layer="last",
            dimensions=width or 0,
            items=items,
        )
    except Exception as exc:
        logger.exception("Token states failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token states failed: {exc}",
        ) from exc
