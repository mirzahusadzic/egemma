import logging
from contextlib import asynccontextmanager
from enum import Enum
from functools import lru_cache
from typing import Annotated, List, Optional

import fastapi
import fastapi_swagger_dark as fsd
from fastapi import Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .config import settings
from .embedding import SentenceTransformerWrapper
from .summarization import EXTENSION_TO_LANGUAGE, SummarizationModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define security scheme
security = HTTPBearer()


async def get_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if settings.WORKBENCH_API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: WORKBENCH_API_KEY not set.",
        )
    if credentials.credentials != settings.WORKBENCH_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class EmbeddingDimensions(int, Enum):
    DIM_128 = 128
    DIM_256 = 256
    DIM_512 = 512
    DIM_768 = 768


embedding_model_wrapper = SentenceTransformerWrapper()
summarization_model_wrapper: SummarizationModelWrapper | None = None


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Loads the models on application startup.
    """
    global summarization_model_wrapper
    embedding_model_wrapper.load_model()
    if settings.SUMMARY_ENABLED:
        summarization_model_wrapper = SummarizationModelWrapper()
        summarization_model_wrapper.load_model()
    yield


app = fastapi.FastAPI(
    title="Code Intelligence API",
    description="API for embedding and summarizing code using Google's Gemma models.",
    version="0.2.0",
    docs_url=None,
    lifespan=lifespan,
)


@lru_cache(maxsize=128)
def cached_encode(text: str):
    return embedding_model_wrapper.encode(text)


class TextInput(BaseModel):
    text: str


async def swagger_ui_html(
    request: fastapi.Request,
) -> fastapi.responses.HTMLResponse:
    return fsd.get_swagger_ui_html(request)


app.get("/docs")(swagger_ui_html)
app.get("/dark_theme.css", include_in_schema=False, name="dark_theme")(
    fsd.dark_swagger_theme
)

DEFAULT_DIMENSIONS_QUERY = fastapi.Query(
    [EmbeddingDimensions.DIM_128], description="Select embedding dimensions"
)


@app.post(
    "/embed",
    summary="Embed Text",
    description="Embeds a given text using the Gemma embedding 300m model "
    "with Matryoshka support.",
    dependencies=[Depends(get_api_key)],
)
async def embed(
    input: TextInput,
    dimensions: Optional[List[EmbeddingDimensions]] = DEFAULT_DIMENSIONS_QUERY,
):
    if dimensions is None:
        dimensions = [EmbeddingDimensions.DIM_128]
    try:
        embedding = await run_in_threadpool(cached_encode, input.text)

        if not dimensions:
            return {"embedding": embedding.tolist()}

        result = {}
        for dim in dimensions:
            dim_value = dim.value
            if dim_value in [128, 256, 512, 768]:
                result[f"embedding_{dim_value}d"] = embedding[:dim_value].tolist()

        return result
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/summarize",
    summary="Summarize Code or Markdown File",
    description="Upload a code or Markdown file and get a Markdown-formatted summary.",
    dependencies=[Depends(get_api_key)],
)
async def summarize(
    file: Annotated[UploadFile, File(...)],
    max_tokens: Optional[int] = Query(
        default=None, description="Maximum number of tokens for the summary."
    ),
    temperature: Optional[float] = Query(
        default=None, description="Temperature for the summary generation."
    ),
):
    if not settings.SUMMARY_ENABLED or summarization_model_wrapper is None:
        raise HTTPException(
            status_code=404, detail="Summarization feature is disabled."
        )
    try:
        code_bytes = await file.read()
        content = code_bytes.decode("utf-8")

        ext = file.filename.split(".")[-1].lower()
        language = EXTENSION_TO_LANGUAGE.get(ext, "code")

        # Use a wrapper to pass keyword arguments to the threadpool
        def do_summarize():
            return summarization_model_wrapper.summarize(
                content,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        summary = await run_in_threadpool(do_summarize)

        # Wrap in Markdown formatting if it's a Markdown file
        if language.lower() == "markdown":
            summary = f"# Summary of Markdown File\n\n{summary}"

        return {"language": language, "summary": summary}
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
