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
from .config import get_conversations_dir, settings
from .conversations import ConversationManager
from .embedding import SentenceTransformerWrapper
from .openai_compat import (
    ModelInfo,
    ModelsResponse,
)
from .responses import (
    Response,
    ResponseUsage,
    convert_chat_response_to_response,
    convert_tools_to_chat_format,
    create_output_item_added_event,
    create_reasoning_delta_event,
    create_response_created_event,
    create_response_done_event,
    create_text_delta_event,
    generate_call_id,
    generate_message_id,
    generate_reasoning_id,
    generate_response_id,
    parse_input_to_messages,
)
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
conversation_manager: ConversationManager | None = None


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Loads the models on application startup and validates configuration.
    """
    global summarization_model_wrapper, chat_model_wrapper, conversation_manager

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

    # Initialize conversation manager (Conversations API)
    # Store conversations in ~/.egemma/{model-name}/conversations
    conv_dir = get_conversations_dir(settings.CHAT_MODEL_NAME)
    conversation_manager = ConversationManager(conversations_dir=conv_dir)
    logger.info(f"Conversation manager initialized: {conv_dir}")

    yield


app = fastapi.FastAPI(
    title="eGemma API",
    description=(
        "Local AI workbench for embeddings, summarization, and chat completions. "
        "Supports Gemma (embeddings/summarization), Gemini API, and GPT-OSS-20B (chat)."
    ),
    version="0.3.0",
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
    logger.debug(
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
    logger.debug(
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
# OpenAI-Compatible Models Endpoint
# =============================================================================


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
# OpenAI Conversations API
# =============================================================================


@app.post(
    "/v1/conversations",
    summary="Create Conversation",
    description="Create a new conversation.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def create_conversation(
    request: Request,
):
    """Create a new conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    # Parse optional metadata from request body
    metadata = None
    try:
        body = await request.json()
        metadata = body.get("metadata")
    except Exception:
        pass  # No body or invalid JSON is OK

    conv = conversation_manager.create(metadata)
    return conv.to_dict()


@app.get(
    "/v1/conversations",
    summary="List Conversations",
    description="List all conversations.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
):
    """List all conversations."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    convs = conversation_manager.list(limit)
    return {"object": "list", "data": convs, "has_more": False}


@app.get(
    "/v1/conversations/{conversation_id}",
    summary="Get Conversation",
    description="Get a conversation by ID.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    conv = conversation_manager.get(conversation_id)
    if conv is None:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")
    return conv.to_dict()


@app.delete(
    "/v1/conversations/{conversation_id}",
    summary="Delete Conversation",
    description="Delete a conversation.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    deleted = conversation_manager.delete(conversation_id)
    if not deleted:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")
    return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}


@app.get(
    "/v1/conversations/{conversation_id}/items",
    summary="Get Conversation Items",
    description="Get items (messages) from a conversation.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def get_conversation_items(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=1000),
    order: str = Query("asc", pattern="^(asc|desc)$"),
):
    """Get items from a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    conv = conversation_manager.get(conversation_id)
    if conv is None:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")

    items = conversation_manager.get_items(conversation_id, limit, order)
    return {"object": "list", "data": items, "has_more": False}


@app.post(
    "/v1/conversations/{conversation_id}/items",
    summary="Add Conversation Items",
    description="Add items (messages) to a conversation.",
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)],
)
async def add_conversation_items(
    conversation_id: str,
    request: Request,
):
    """Add items to a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON body: {e}") from e

    items = body.get("items", [])
    if not items:
        raise HTTPException(400, "No items provided")

    try:
        added = conversation_manager.add_items(conversation_id, items)
        return {"object": "list", "data": added}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


# =============================================================================
# OpenAI Responses API
# =============================================================================


@app.post(
    "/v1/responses",
    summary="Create Response (OpenAI Responses API)",
    description=(
        "Generate a model response using the Responses API format. "
        "Supports stateful conversations, tool calling, and streaming."
    ),
    tags=["Responses API"],
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
async def create_response(
    request: Request,
):
    """
    OpenAI Responses API endpoint.

    Creates a model response given input. Supports:
    - Simple string input or structured input items
    - System instructions
    - Tool definitions and tool calling
    - Stateful conversations via previous_response_id
    - Streaming via SSE

    This is the primary API for the OpenAI Agents SDK.
    """
    if chat_model_wrapper is None or not chat_model_wrapper.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Chat model not loaded. Set CHAT_MODEL_ENABLED=true in .env",
        )

    # Parse request body
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON body: {e}") from e

    # Extract parameters
    input_data = body.get("input")
    model = body.get("model", settings.CHAT_MODEL_NAME)
    instructions = body.get("instructions")
    max_output_tokens = body.get("max_output_tokens", settings.CHAT_MAX_TOKENS)
    temperature = body.get("temperature", settings.CHAT_TEMPERATURE)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    stream = body.get("stream", False)
    # Support both 'previous_response_id' and 'conversation' fields for history loading
    # OpenAI SDK sends both fields - use whichever is provided
    previous_response_id = body.get("previous_response_id") or body.get("conversation")

    # Convert input to chat messages
    messages = parse_input_to_messages(input_data, instructions)

    # If previous_response_id/conversation provided, load conversation history
    # The previous_response_id may be a conversation_id from the Conversations API
    if previous_response_id and conversation_manager:
        try:
            # Load conversation history and prepend to current messages
            history = conversation_manager.get_items_as_messages(previous_response_id)
            if history:
                # Prepend history before current input (after system msg if present)
                # System message is typically first in messages list
                if messages and messages[0].get("role") == "system":
                    # Keep system message first, then history, then current input
                    system_msg = messages[0]
                    current_input = messages[1:] if len(messages) > 1 else []
                    messages = [system_msg] + history + current_input
                else:
                    # No system message, just prepend history
                    messages = history + messages
                logger.debug(
                    f"Loaded {len(history)} history items for "
                    f"conversation {previous_response_id}"
                )
        except Exception as e:
            # Graceful degradation - continue without history if lookup fails
            logger.warning(f"Failed to load conversation history: {e}")

    # Convert tools to chat format
    chat_tools = convert_tools_to_chat_format(tools)

    # Generate response ID
    response_id = generate_response_id()

    try:
        if stream:
            # Streaming response
            from fastapi.responses import StreamingResponse

            async def generate_stream():
                # Create initial response object
                initial_response = Response(
                    id=response_id,
                    model=model,
                    status="in_progress",
                    instructions=instructions,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    previous_response_id=previous_response_id,
                )

                # Send response.created event
                yield create_response_created_event(initial_response)

                # Track output for final response
                collected_content = []
                collected_thinking = []
                message_item_id = generate_message_id()
                reasoning_item_id = generate_reasoning_id()
                usage = ResponseUsage()
                in_tool_call = False  # Track if we're streaming a tool call
                json_brace_depth = 0  # Track JSON object depth for standalone JSON

                # Log request for debugging (use DEBUG level to avoid console spam)
                logger.debug(f"[STREAM] Request to model: {len(messages)} messages")
                for i, msg in enumerate(messages):
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                    tool_call_id = msg.get("tool_call_id", "")
                    names = [
                        tc.get("function", {}).get("name", "?") for tc in tool_calls
                    ]
                    tc_info = f" [calls: {names}]" if tool_calls else ""
                    tc_id_info = f" [for: {tool_call_id}]" if tool_call_id else ""
                    logger.debug(f"  [{i}] {role}{tc_info}{tc_id_info}: {content}")

                # Call chat model with streaming
                stream_response = await run_in_threadpool(
                    chat_model_wrapper.create_completion,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    tools=chat_tools,
                    tool_choice=tool_choice,
                    stream=True,
                    include_thinking=True,  # o-series reasoning
                )

                for chunk in stream_response:
                    # Check for client disconnect
                    if await request.is_disconnected():
                        break

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})

                        # Handle reasoning/thinking (GPT-OSS Harmony -> OpenAI o-series)
                        thinking = delta.get("thinking", "")
                        if thinking:
                            collected_thinking.append(thinking)
                            yield create_reasoning_delta_event(
                                response_id, thinking, reasoning_item_id
                            )

                        # Handle text content
                        content = delta.get("content", "")
                        if content:
                            # Always collect content for tool call parsing
                            collected_content.append(content)

                            # FIX: Don't stream tool call content as text
                            # Tool calls span multiple chunks, so we need state tracking

                            # Check if we're currently in a tool call BEFORE processing
                            was_in_tool_call = in_tool_call

                            # Detect start of Harmony format tool call
                            if "<|channel|>commentary" in content:
                                in_tool_call = True

                            # Detect end of Harmony format tool call
                            if "<|end|>" in content or "<|call|>" in content:
                                in_tool_call = False

                            # Detect standalone JSON tool calls
                            # Check for tool call patterns and track JSON depth
                            if not in_tool_call and (
                                '{"command"' in content
                                or '{"file_path"' in content
                                or '{"path"' in content
                                or '{"pattern"' in content
                                or '{"url"' in content
                                or '{"function"' in content
                                or '{"tool"' in content
                            ):
                                # Looks like start of standalone JSON tool call
                                in_tool_call = True
                                was_in_tool_call = True  # Suppress entire chunk
                                json_brace_depth = 0

                            # Track brace depth for JSON objects
                            if json_brace_depth >= 0:
                                for char in content:
                                    if char == "{":
                                        json_brace_depth += 1
                                    elif char == "}":
                                        json_brace_depth -= 1
                                        if json_brace_depth == 0 and in_tool_call:
                                            # JSON object complete
                                            in_tool_call = False

                            # Only stream if we were NOT in a tool call at the start
                            if not was_in_tool_call and not in_tool_call:
                                yield create_text_delta_event(
                                    response_id, content, message_item_id
                                )

                        # Note: Tool calls are NOT streamed by llama-cpp.
                        # We parse them from collected_content after streaming.

                    # Extract usage if present
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage.input_tokens = chunk_usage.get("prompt_tokens", 0)
                        usage.output_tokens = chunk_usage.get("completion_tokens", 0)
                        usage.total_tokens = chunk_usage.get("total_tokens", 0)

                # Build final response
                output = []
                output_text = "".join(collected_content)
                output_index = 0

                # Include reasoning first (model thinks, then responds)
                # SDK reads "summary" field, converts to "content" internally
                reasoning_text = "".join(collected_thinking)
                if reasoning_text:
                    summary_item = {"type": "summary_text", "text": reasoning_text}
                    reasoning_item = {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [summary_item],
                    }
                    output.append(reasoning_item)
                    # Emit output_item.added event for SDK
                    yield create_output_item_added_event(
                        response_id, reasoning_item, output_index
                    )
                    output_index += 1

                # Parse tool calls from collected content (streaming doesn't
                # emit tool_calls, so we parse from text like batch mode)
                parsed_tool_calls = []
                if tools and output_text:
                    parsed_tool_calls = (
                        chat_model_wrapper._parse_tool_calls(output_text) or []
                    )

                # If we have tool calls, add them and clear text output
                if parsed_tool_calls:
                    for tc in parsed_tool_calls:
                        func = tc.get("function", {})
                        fc_item = {
                            "id": generate_message_id(),
                            "type": "function_call",
                            "call_id": tc.get("id", generate_call_id()),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                            "status": "completed",
                        }
                        output.append(fc_item)
                        # Emit output_item.added event for SDK
                        yield create_output_item_added_event(
                            response_id, fc_item, output_index
                        )
                        output_index += 1
                    # Clear text output when we have tool calls (like batch mode)
                    output_text = ""
                elif output_text:
                    # Only add message output if no tool calls
                    message_item = {
                        "id": message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": output_text}],
                    }
                    output.append(message_item)
                    # Emit output_item.added event for SDK
                    yield create_output_item_added_event(
                        response_id, message_item, output_index
                    )
                    output_index += 1

                # Create final response
                final_response = Response(
                    id=response_id,
                    model=model,
                    output=output,
                    output_text=output_text,
                    status="completed",
                    instructions=instructions,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    previous_response_id=previous_response_id,
                    usage=usage,
                )

                # Send response.completed event
                yield create_response_done_event(final_response)

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        else:
            # Non-streaming response
            # Log request for debugging (use DEBUG level to avoid console spam)
            logger.debug(f"[BATCH] Request to model: {len(messages)} messages")
            for i, msg in enumerate(messages):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                tool_call_id = msg.get("tool_call_id", "")
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                tc_info = f" [calls: {names}]" if tool_calls else ""
                tc_id_info = f" [for: {tool_call_id}]" if tool_call_id else ""
                logger.debug(f"  [{i}] {role}{tc_info}{tc_id_info}: {content}")

            chat_response = await run_in_threadpool(
                chat_model_wrapper.create_completion,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                tools=chat_tools,
                tool_choice=tool_choice,
                stream=False,
                include_thinking=True,  # o-series reasoning
            )

            # Convert to Responses API format
            response = convert_chat_response_to_response(
                chat_response=chat_response,
                response_id=response_id,
                model=model,
                instructions=instructions,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                previous_response_id=previous_response_id,
            )

            return response.to_dict()

    except Exception as e:
        logger.error(f"Responses API error: {e}", exc_info=True)
        # Return error in Responses API format
        error_response = Response(
            id=response_id,
            model=model,
            status="failed",
            error={"type": "server_error", "message": str(e)},
        )
        raise HTTPException(status_code=500, detail=error_response.to_dict()) from e


@app.get("/", tags=["Root"])
async def read_root():
    """
    Redirects the root URL to the API documentation.
    """
    return fastapi.responses.RedirectResponse(url="/docs")
