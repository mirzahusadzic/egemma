import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import fastapi
import fastapi_swagger_dark as fsd
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .api.openai.compat import ModelInfo, ModelsResponse
from .api.openai.conversations import ConversationManager
from .config import get_conversations_dir, settings
from .models.embedding import SentenceTransformerWrapper
from .models.llm import ChatModelWrapper
from .models.summarization import SummarizationModelWrapper

# Import routers
from .routers import ast_parser, conversations, embed, responses, summarize

# Configure logging with explicit format to ensure visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# Ensure our module loggers are set to INFO
logging.getLogger("src.models.llm").setLevel(logging.INFO)
logging.getLogger("src.streaming.handler").setLevel(logging.INFO)

# Log startup message to confirm logging is working
logger.info("=" * 80)
logger.info("ℹ️  INFO LOGGING ENABLED - Reduced verbosity")
logger.info("=" * 80)


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
    if credentials.credentials != settings.WORKBENCH_API_KEY:
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

    # Load embedding model
    embed.embedding_model_wrapper = SentenceTransformerWrapper()
    embed.embedding_model_wrapper.load_model()

    # Load summarization model
    # Always initialize summarization wrapper (for Gemini API support)
    # but only load local model if SUMMARY_LOCAL_ENABLED is True
    summarization_wrapper = SummarizationModelWrapper()
    if settings.SUMMARY_LOCAL_ENABLED:
        summarization_wrapper.load_local_model()
    # Always try to initialize Gemini client if API key is present
    if settings.GEMINI_API_KEY:
        summarization_wrapper.load_gemini_client()
    summarize.summarization_model_wrapper = summarization_wrapper

    # Load chat model (GPT-OSS-20B) if enabled
    if settings.CHAT_MODEL_ENABLED:
        chat_wrapper = ChatModelWrapper()
        chat_wrapper.load_model()
        responses.chat_model_wrapper = chat_wrapper
    else:
        responses.chat_model_wrapper = None

    # Initialize conversation manager (Conversations API)
    # Store conversations in ~/.egemma/{model-name}/conversations
    conv_dir = get_conversations_dir(settings.CHAT_MODEL_NAME)
    conv_manager = ConversationManager(conversations_dir=conv_dir)
    conversations.conversation_manager = conv_manager
    responses.conversation_manager = conv_manager
    logger.info(f"Conversation manager initialized: {conv_dir}")

    yield


app = fastapi.FastAPI(
    title="Code Intelligence API (eGemma)",
    description=(
        "Local AI workbench for embeddings, summarization, and OpenAI Agent "
        "SDK–compatible workflows. Supports Gemma models, GPT-OSS tool-calling, "
        "hardware acceleration, caching, and rate limiting, plus cloud-offloaded, "
        "persona-driven summarization through Gemini."
    ),
    version="0.4.0",
    docs_url=None,
    lifespan=lifespan,
)


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
        "summarize": {
            "calls": settings.SUMMARIZE_RATE_LIMIT_CALLS,
            "seconds": settings.SUMMARIZE_RATE_LIMIT_SECONDS,
        },
        "chat": {
            "calls": settings.CHAT_RATE_LIMIT_CALLS,
            "seconds": settings.CHAT_RATE_LIMIT_SECONDS,
        },
        "conversation": {
            "calls": settings.CONVERSATION_RATE_LIMIT_CALLS,
            "seconds": settings.CONVERSATION_RATE_LIMIT_SECONDS,
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
        "status": (
            "loaded"
            if embed.embedding_model_wrapper
            and embed.embedding_model_wrapper.model is not None
            else "not_loaded"
        ),
    }
    response["embedding_model"] = embedding_status

    # Check local summarization model if enabled
    if settings.SUMMARY_LOCAL_ENABLED:
        summarization_status = {
            "name": settings.SUMMARY_MODEL_NAME,
            "basename": settings.SUMMARY_MODEL_BASENAME,
            "status": (
                "loaded"
                if summarize.summarization_model_wrapper
                and summarize.summarization_model_wrapper.model is not None
                else "not_loaded"
            ),
        }
        response["local_summarization_model"] = summarization_status
    else:
        response["local_summarization_model"] = {"status": "disabled"}

    # Check Gemini API status
    gemini_status = {
        "api_key_set": bool(settings.GEMINI_API_KEY),
        "default_model": settings.GEMINI_DEFAULT_MODEL,
    }
    response["gemini_api"] = gemini_status

    # Check chat model status (GPT-OSS-20B)
    if settings.CHAT_MODEL_ENABLED:
        chat_status = {
            "name": settings.CHAT_MODEL_NAME,
            "path": settings.CHAT_MODEL_PATH,
            "status": (
                "loaded"
                if responses.chat_model_wrapper
                and responses.chat_model_wrapper.is_loaded
                else "not_loaded"
            ),
            "context_length": settings.CHAT_N_CTX,
            "supports_tools": True,
        }
        response["chat_model"] = chat_status
    else:
        response["chat_model"] = {"status": "disabled"}

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
# OpenAI-Compatible Models Endpoint
# =============================================================================


@app.get(
    "/v1/models",
    summary="List Models (OpenAI-compatible)",
    description="List available models for chat completions.",
    tags=["OpenAI API"],
    response_model=ModelsResponse,
    dependencies=[Depends(get_api_key)],
)
async def list_models():
    """List available models."""
    import time

    models = []

    # Add chat model if enabled
    if (
        settings.CHAT_MODEL_ENABLED
        and responses.chat_model_wrapper
        and responses.chat_model_wrapper.is_loaded
    ):
        models.append(
            ModelInfo(
                id=settings.CHAT_MODEL_NAME,
                created=int(time.time()),
                owned_by="egemma",
            )
        )

    return ModelsResponse(data=models)


# =============================================================================
# Include Routers
# =============================================================================

# Embedding router
app.include_router(embed.router, dependencies=[Depends(get_api_key)])

# Summarization router
app.include_router(summarize.router, dependencies=[Depends(get_api_key)])

# AST parser router
app.include_router(ast_parser.router, dependencies=[Depends(get_api_key)])

# Conversations router (OpenAI API)
app.include_router(conversations.router, dependencies=[Depends(get_api_key)])

# Responses router (OpenAI Responses API)
app.include_router(responses.router, dependencies=[Depends(get_api_key)])


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
