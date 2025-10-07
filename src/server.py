import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, List, Optional

import fastapi
import fastapi_swagger_dark as fsd
import torch
from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Conditionally load environment variables from .env file
if os.getenv("WORKBENCH_APP_ENV") != "production":
    from dotenv import load_dotenv

    load_dotenv()

# Retrieve API key from environment
WORKBENCH_API_KEY = os.getenv("WORKBENCH_API_KEY")

# Define security scheme
security = HTTPBearer()


async def get_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if WORKBENCH_API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: WORKBENCH_API_KEY not set.",
        )
    if credentials.credentials != WORKBENCH_API_KEY:
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


class SentenceTransformerWrapper:
    def __init__(self):
        self.model: "Optional[SentenceTransformer]" = None

    @staticmethod
    def _get_optimal_device() -> str:
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using CUDA device.")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("MPS is available. Using MPS device.")
            return "mps"
        else:
            logger.info("No GPU acceleration available. Using CPU device.")
            return "cpu"

    def load_model(self):
        from sentence_transformers import SentenceTransformer

        device = self._get_optimal_device()
        self.model = SentenceTransformer("google/embeddinggemma-300m", device=device)

    def encode(self, text: str):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.encode(text)


model_wrapper = SentenceTransformerWrapper()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Loads the SentenceTransformer model on application startup.
    """
    model_wrapper.load_model()
    yield


app = fastapi.FastAPI(
    title="Embedding API",
    description="API for embedding text using the Gemma embedding 300m model "
    "with Matryoshka support.",
    version="0.1.0",
    docs_url=None,
    lifespan=lifespan,
)


@lru_cache(maxsize=128)
def cached_encode(text: str):
    return model_wrapper.encode(text)


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


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
