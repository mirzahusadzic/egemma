import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from functools import lru_cache
from typing import Annotated, List

import fastapi
import fastapi_swagger_dark as fsd
from fastapi import (
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .chat import ChatModelWrapper
from .config import settings
from .embedding import SentenceTransformerWrapper
from .openai_compat import (
    ChatCompletionRequest,
    ModelInfo,
    ModelsResponse,
)
from .session import SessionManager
from .summarization import SummarizationModelWrapper
from .util import condense_log, is_likely_binary
from .util.rate_limiter import get_in_memory_rate_limiter

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
            detail="Server configuration error: API key not set.",
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


embedding_model_wrapper = SentenceTransformerWrapper()
summarization_model_wrapper: SummarizationModelWrapper | None = None
chat_model_wrapper: ChatModelWrapper | None = None
session_manager: SessionManager | None = None


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Loads the models on application startup and validates configuration.
    """
    global summarization_model_wrapper, chat_model_wrapper, session_manager

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

    # Load models
    embedding_model_wrapper.load_model()
    # Always initialize summarization wrapper (for Gemini API support)
    # but only load local model if SUMMARY_LOCAL_ENABLED is True
    summarization_model_wrapper = SummarizationModelWrapper()
    if settings.SUMMARY_LOCAL_ENABLED:
        summarization_model_wrapper.load_local_model()
    # Always try to initialize Gemini client if API key is present
    if settings.GEMINI_API_KEY:
        summarization_model_wrapper.load_gemini_client()

    # Load chat model (GPT-OSS-20B) if enabled
    if settings.CHAT_MODEL_ENABLED:
        chat_model_wrapper = ChatModelWrapper()
        chat_model_wrapper.load_model()

        # Initialize session manager with model for tokenization
        session_manager = SessionManager(
            model=chat_model_wrapper.model,
            max_context=settings.CHAT_N_CTX,
            sessions_dir="sessions",
        )
        logger.info("Session manager initialized")

    yield


app = fastapi.FastAPI(
    title="Code Intelligence API",
    description="API for embedding and summarizing code using Google's Gemma models.",
    version="0.2.0",
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
            "loaded" if embedding_model_wrapper.model is not None else "not_loaded"
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
                if summarization_model_wrapper
                and summarization_model_wrapper.model is not None
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
                if chat_model_wrapper and chat_model_wrapper.is_loaded
                else "not_loaded"
            ),
            "context_length": settings.CHAT_N_CTX,
            "supports_tools": True,
        }
        response["chat_model"] = chat_status
    else:
        response["chat_model"] = {"status": "disabled"}

    return fastapi.responses.JSONResponse(content=response)


@lru_cache(maxsize=settings.CACHE_MAX_SIZE)
def cached_encode(text: str, prompt_name: str | None = None, title: str | None = None):
    return embedding_model_wrapper.encode(text, prompt_name=prompt_name, title=title)


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

DEFAULT_PROMPT_NAME_QUERY = fastapi.Query(
    PromptNames.DOCUMENT, description="Select a prompt for the embedding model"
)


@app.post(
    "/embed",
    summary="Embed File Content",
    description=(
        "Embeds the content of an uploaded file using the Gemma embedding 300m model "
        "with Matryoshka support."
    ),
    dependencies=[
        Depends(get_api_key),
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.EMBED_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.EMBED_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def embed(
    file: Annotated[UploadFile, File(..., max_size=settings.MAX_FILE_SIZE_BYTES)],
    dimensions: List[EmbeddingDimensions] | None = DEFAULT_DIMENSIONS_QUERY,
    prompt_name: PromptNames | None = DEFAULT_PROMPT_NAME_QUERY,
    title: str | None = Query(
        None, description="Optional title for the embedded content."
    ),
):
    if dimensions is None:
        dimensions = [EmbeddingDimensions.DIM_128]
    try:
        file_content_bytes = await file.read()
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
    except Exception as e:
        logger.error(f"Error during embedding: {e}", exc_info=True)
        raise fastapi.HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        ) from e


@app.post(
    "/summarize",
    summary="Summarize Code, Markdown, or Log File",
    description="Upload a code, Markdown, or log file and get a summary. "
    "Log files are automatically condensed before summarization.",
    dependencies=[
        Depends(get_api_key),
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.SUMMARIZE_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.SUMMARIZE_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def summarize(
    request: fastapi.Request,  # Add request object to access headers
    file: Annotated[UploadFile, File(..., max_size=settings.MAX_FILE_SIZE_BYTES)],
    max_tokens: int | None = fastapi.Form(
        default=None, description="Maximum number of tokens for the summary."
    ),
    temperature: float | None = fastapi.Form(
        default=None, description="Temperature for the summary generation."
    ),
    persona: str | None = fastapi.Form(
        default=None,
        description=(
            "Persona to use for summarization "
            "(e.g., 'developer', 'assistant', 'security_validator')."
        ),
    ),
    model_name: str | None = fastapi.Form(
        default=None,
        description=(
            "Name of the model to use for summarization (e.g., 'gemini-2.5-flash')."
        ),
    ),
    enable_safety: bool | None = fastapi.Form(
        default=False,
        description=(
            "Enable Gemini safety settings for content filtering "
            "(only applies to Gemini models)."
        ),
    ),
):
    logger.info(
        f"Received /summarize request. "
        f"Content-Type: {request.headers.get('content-type')}"
    )
    if summarization_model_wrapper is None:
        raise HTTPException(
            status_code=500, detail="Summarization service not initialized."
        )

    # Check if requesting local model when it's disabled
    if model_name is None or not model_name.startswith("gemini"):
        if not settings.SUMMARY_LOCAL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Local summarization is disabled. Please specify a Gemini "
                    "model (e.g., model_name='gemini-2.5-flash')."
                ),
            )
    try:
        code_bytes = await file.read()
        if is_likely_binary(code_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file appears to be binary and cannot be summarized.",
            )
        content = code_bytes.decode("utf-8")

        ext = file.filename.split(".")[-1].lower()
        language = settings.EXTENSION_TO_LANGUAGE.get(ext, "code")

        # Condense log files before summarization
        if language == "Log File":
            content = condense_log(content)

        # Determine persona name, with defaults based on language type
        if persona is None:
            if language.lower() == "markdown":
                persona_name = "assistant"
            else:
                persona_name = "developer"
        else:
            persona_name = persona

        # Use a wrapper to pass keyword arguments to the threadpool
        def do_summarize():
            return summarization_model_wrapper.summarize(
                content,
                language=language,
                persona_name=persona_name,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=model_name,
                enable_safety=enable_safety,
            )

        summary = await run_in_threadpool(do_summarize)

        return {"language": language, "summary": summary}
    except HTTPException as e:
        logger.error(
            f"Summarization failed with HTTP Exception: {e.detail} "
            f"(status: {e.status_code}) (file: {file.filename})",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"An internal server error occurred during summarization: {e} "
            f"(file: {file.filename})",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        ) from e


@app.post("/parse-ast", dependencies=[Depends(get_api_key)])
async def parse_ast(
    request: fastapi.Request,  # Add request object to access headers
    file: UploadFile,
    language: str = fastapi.Form(...),
):
    logger.info(
        f"Received /parse-ast request. "
        f"Content-Type: {request.headers.get('content-type')}"
    )
    """
    Deterministic AST parsing endpoint for non-native languages.
    Currently supports Python via the ast module.
    """
    if language != "python":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Language '{language}' not supported. "
                "Currently only 'python' is available."
            ),
        )

    content = await file.read()
    code = content.decode("utf-8")

    from src.util.ast_parser import parse_code_to_ast

    parsed_ast = parse_code_to_ast(code, language)
    return parsed_ast


# =============================================================================
# OpenAI-Compatible Chat Endpoints
# =============================================================================


@app.post(
    "/v1/chat/completions",
    summary="Chat Completions (OpenAI-compatible)",
    description="GPT-OSS-20B chat. Supports streaming, tool calling, and sessions.",
    tags=["OpenAI API"],
    dependencies=[
        Depends(get_api_key),
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CHAT_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CHAT_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """
    OpenAI-compatible chat completions endpoint.

    Optional session support via X-Session-Id header:
    - If provided, prepends stored messages from session
    - Stores user/assistant messages after completion
    - Without header, operates stateless (standard OpenAI behavior)
    """
    if chat_model_wrapper is None or not chat_model_wrapper.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Chat model not loaded. Set CHAT_MODEL_ENABLED=true in .env",
        )

    # Convert Pydantic messages to dicts
    request_messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    # Session handling: prepend stored messages if session specified
    session = None
    if x_session_id and session_manager:
        session = session_manager.get_session(x_session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {x_session_id}",
            )
        # Prepend session history to request messages
        stored_messages = session_manager.get_messages()
        messages = stored_messages + request_messages
        logger.debug(
            f"Session {x_session_id}: {len(stored_messages)} stored + "
            f"{len(request_messages)} new messages"
        )
    else:
        messages = request_messages

    # Convert tools if provided
    tools = None
    if request.tools:
        tools = [tool.model_dump() for tool in request.tools]

    try:
        if request.stream:
            # Streaming response (SSE)
            from fastapi.responses import StreamingResponse

            collected_content = []

            async def generate_stream():
                stream = await run_in_threadpool(
                    chat_model_wrapper.create_completion,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=tools,
                    tool_choice=request.tool_choice,
                    stream=True,
                    include_thinking=request.include_thinking,
                )
                import json

                for chunk in stream:
                    # Collect content for session storage
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        collected_content.append(delta["content"])
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

                # Store messages in session after streaming completes
                if session and session_manager:
                    # Store user messages from request
                    for msg in request_messages:
                        session_manager.add_message(msg)
                    # Store assistant response
                    if collected_content:
                        session_manager.add_message(
                            {
                                "role": "assistant",
                                "content": "".join(collected_content),
                            }
                        )

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-Id": x_session_id or "",
                },
            )
        else:
            # Non-streaming response
            response = await run_in_threadpool(
                chat_model_wrapper.create_completion,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=tools,
                tool_choice=request.tool_choice,
                stream=False,
                include_thinking=request.include_thinking,
            )

            # Store messages in session
            if session and session_manager:
                # Store user messages from request
                for msg in request_messages:
                    session_manager.add_message(msg)
                # Store assistant response
                assistant_msg = response.get("choices", [{}])[0].get("message", {})
                if assistant_msg.get("content"):
                    session_manager.add_message(
                        {
                            "role": "assistant",
                            "content": assistant_msg["content"],
                        }
                    )

            return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/v1/models",
    summary="List Models (OpenAI-compatible)",
    description="List available models for chat completions.",
    tags=["OpenAI API"],
    response_model=ModelsResponse,
)
async def list_models():
    """List available models."""
    import time

    models = []

    # Add chat model if enabled
    if (
        settings.CHAT_MODEL_ENABLED
        and chat_model_wrapper
        and chat_model_wrapper.is_loaded
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
# Session Management Endpoints
# =============================================================================


@app.post(
    "/v1/sessions",
    summary="Create Session",
    description="Create a new chat session. Invalidates any existing session.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def create_session():
    """Create a new session, invalidating any existing one."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Enable chat model.",
        )

    session = session_manager.create_session()
    return {
        "session_id": session.session_id,
        "max_context": session.max_context,
        "created_at": session.created_at,
    }


@app.get(
    "/v1/sessions/current",
    summary="Get Current Session",
    description="Get current active session info and token usage.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def get_current_session():
    """Get current session stats."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized.",
        )

    session = session_manager.get_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")

    return session.to_stats()


@app.post(
    "/v1/sessions/{session_id}/load",
    summary="Load Session",
    description="Load/resume a previously saved session by ID.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def load_session(session_id: str):
    """Load an existing session by ID."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized.",
        )

    session = session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    return session.to_stats()


@app.get(
    "/v1/sessions/{session_id}/messages",
    summary="Get Session Messages",
    description="Get all messages from a session.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def get_session_messages(session_id: str):
    """Get all messages stored in a session."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized.",
        )

    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    return {
        "session_id": session.session_id,
        "message_count": len(session.messages),
        "token_count": session.token_count,
        "messages": session.messages,
    }


@app.get(
    "/v1/sessions",
    summary="List Sessions",
    description="List all saved sessions.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def list_sessions():
    """List all saved sessions."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized.",
        )

    return {"sessions": session_manager.list_sessions()}


@app.post(
    "/v1/sessions/count-tokens",
    summary="Count Tokens",
    description="Count tokens in provided text using the model's tokenizer.",
    tags=["Sessions"],
    dependencies=[Depends(get_api_key)],
)
async def count_tokens(text: str = Query(..., description="Text to count tokens for")):
    """Count tokens in text."""
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized.",
        )

    token_count = session_manager.count_tokens(text)
    return {"text_length": len(text), "token_count": token_count}


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
