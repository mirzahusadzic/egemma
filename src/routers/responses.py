"""
OpenAI Responses API router.
Handles /v1/responses endpoint for chat completions with tool calling.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from ..api.openai.conversations import ConversationManager
from ..api.openai.responses import (
    Response,
    convert_chat_response_to_response,
    convert_tools_to_chat_format,
    generate_response_id,
    parse_input_to_messages,
)
from ..config import settings
from ..models.llm import ChatModelWrapper
from ..streaming.handler import generate_streaming_response
from ..util.rate_limiter import get_in_memory_rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared model wrappers (initialized in server.py)
chat_model_wrapper: ChatModelWrapper | None = None
conversation_manager: ConversationManager | None = None

# Create router
router = APIRouter(tags=["Responses API"], prefix="/v1")


@router.post(
    "/responses",
    summary="Create Response (OpenAI Responses API)",
    description=(
        "Generate a model response using the Responses API format. "
        "Supports stateful conversations, tool calling, and streaming."
    ),
    dependencies=[
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
            # Streaming response using streaming handler
            return StreamingResponse(
                generate_streaming_response(
                    request=request,
                    chat_model_wrapper=chat_model_wrapper,
                    response_id=response_id,
                    model=model,
                    messages=messages,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    chat_tools=chat_tools,
                    tool_choice=tool_choice,
                    instructions=instructions,
                    previous_response_id=previous_response_id,
                ),
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
