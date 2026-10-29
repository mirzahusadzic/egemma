import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

import fastapi
import fastapi_swagger_dark as fsd
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .models.amrbart import AMRBartModelWrapper
from .models.embedding import SentenceTransformerWrapper
from .models.summarization import SummarizationModelWrapper

# Import routers
from .routers import amr, ast_parser, corpus, embed, summarize
from .util.body_limit import BodyLimit

# Configure logging with explicit format to ensure visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# PENMAN logs every decoded graph at INFO; keep model payloads in API responses,
# not in the server console.
logging.getLogger("penman").setLevel(logging.WARNING)

# Log startup message to confirm logging is working
logger.debug("=" * 80)
logger.debug("🐛 DEBUG LOGGING ENABLED - Full verbosity")
logger.debug("=" * 80)


# Define security scheme
security = HTTPBearer()


async def get_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if settings.WORKBENCH_API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API key not set.",
        )
    if not secrets.compare_digest(
        credentials.credentials.encode(), settings.WORKBENCH_API_KEY.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Loads the models on application startup and validates configuration.
    """
    # Validate API key configuration
    if not settings.WORKBENCH_API_KEY:
        logger.warning(
            "WORKBENCH_API_KEY is not set. "
            "Endpoints requiring authentication will fail."
        )

    # Validate persona paths
    persona_dir = "personas"
    if not os.path.exists(persona_dir) or not os.path.isdir(persona_dir):
        raise RuntimeError(
            f"Persona directory not found at '{persona_dir}'. Please create it."
        )

    embed.embedding_model_wrapper = SentenceTransformerWrapper()
    embed.embedding_model_wrapper.load_model()

    if settings.AMRBART_MODEL_ENABLED:
        amr_wrapper = AMRBartModelWrapper()
        amr_wrapper.load_model()
        amr.amrbart_model_wrapper = amr_wrapper
    else:
        amr.amrbart_model_wrapper = None

    if settings.SUMMARY_MODEL_ENABLED:
        summarization_wrapper = SummarizationModelWrapper()
        summarization_wrapper.load_model()
        summarize.summarization_model_wrapper = summarization_wrapper
    else:
        summarize.summarization_model_wrapper = None

    yield


app = fastapi.FastAPI(
    title="Code Intelligence API (eGemma)",
    description=(
        "Local inference for embeddings, semantic parsing, and persona-driven "
        "summarization with EmbeddingGemma, AMRBART, and Gemma 4, with hardware "
        "acceleration, caching, and rate limiting."
    ),
    version="0.5.0",
    docs_url=None,
    lifespan=lifespan,
)
# Before anything reads the body: parsing and authentication come after it.
app.add_middleware(BodyLimit)


@app.exception_handler(SyntaxError)
async def syntax_error_exception_handler(request: Request, exc: SyntaxError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Python syntax error: {str(exc)}"},
    )


@app.get("/rate-limits", summary="Rate Limits", tags=["Monitoring"])
async def rate_limits():
    """
    Returns the current rate limit configuration.
    Clients should query this endpoint on startup to configure their rate limiters.
    """
    return {
        "embed": {
            "calls": settings.EMBED_RATE_LIMIT_CALLS,
            "seconds": settings.EMBED_RATE_LIMIT_SECONDS,
        },
        "amr": {
            "calls": settings.AMRBART_RATE_LIMIT_CALLS,
            "seconds": settings.AMRBART_RATE_LIMIT_SECONDS,
        },
        "summarize": {
            "calls": settings.SUMMARIZE_RATE_LIMIT_CALLS,
            "seconds": settings.SUMMARIZE_RATE_LIMIT_SECONDS,
        },
    }


@app.get("/health", summary="Health Check", tags=["Monitoring"])
async def health_check():
    """
    Checks if the server and models are running.
    """

    response = {"status": "ok"}

    # Check embedding model
    embedding_status = {
        "name": settings.EMBEDDING_MODEL_NAME,
        "revision": settings.EMBEDDING_MODEL_REVISION,
        "status": (
            "loaded"
            if embed.embedding_model_wrapper
            and embed.embedding_model_wrapper.model is not None
            else "not_loaded"
        ),
    }
    response["embedding_model"] = embedding_status

    if settings.SUMMARY_MODEL_ENABLED:
        response["summarization_model"] = {
            "name": settings.SUMMARY_MODEL_NAME,
            "basename": settings.SUMMARY_MODEL_BASENAME,
            "revision": settings.SUMMARY_MODEL_REVISION,
            "status": (
                "loaded"
                if summarize.summarization_model_wrapper
                and summarize.summarization_model_wrapper.model is not None
                else "not_loaded"
            ),
        }
    else:
        response["summarization_model"] = {"status": "disabled"}

    if settings.AMRBART_MODEL_ENABLED:
        response["amrbart_model"] = {
            "name": settings.AMRBART_MODEL_NAME,
            "revision": settings.AMRBART_MODEL_REVISION,
            "status": (
                "loaded"
                if amr.amrbart_model_wrapper and amr.amrbart_model_wrapper.is_loaded
                else "not_loaded"
            ),
            "device": (
                amr.amrbart_model_wrapper.device
                if amr.amrbart_model_wrapper
                else "not_loaded"
            ),
        }
    else:
        response["amrbart_model"] = {"status": "disabled"}

    return fastapi.responses.JSONResponse(content=response)


async def swagger_ui_html(
    request: fastapi.Request,
) -> fastapi.responses.HTMLResponse:
    return fsd.get_swagger_ui_html(request)


app.get("/docs")(swagger_ui_html)
app.get("/dark_theme.css", include_in_schema=False, name="dark_theme")(
    fsd.dark_swagger_theme
)


# =============================================================================
# Include Routers
# =============================================================================

# Embedding router
app.include_router(embed.router, dependencies=[Depends(get_api_key)])

# Corpus transformer router
app.include_router(corpus.router, dependencies=[Depends(get_api_key)])

# Abstract Meaning Representation router
app.include_router(amr.router, dependencies=[Depends(get_api_key)])

# Summarization router
app.include_router(summarize.router, dependencies=[Depends(get_api_key)])

# AST parser router
app.include_router(ast_parser.router, dependencies=[Depends(get_api_key)])


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
