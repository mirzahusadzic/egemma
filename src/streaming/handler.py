"""
Streaming response handler for the Responses API.
Handles server-sent events (SSE) for streaming chat completions.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List

from fastapi import Request
from fastapi.concurrency import run_in_threadpool

from ..api.openai.responses import (
    Response,
    ResponseUsage,
    create_output_item_added_event,
    create_reasoning_delta_event,
    create_reasoning_done_event,
    create_response_created_event,
    create_response_done_event,
    create_text_delta_event,
    generate_call_id,
    generate_message_id,
    generate_reasoning_id,
)
from ..models.llm import ChatModelWrapper
from ..util import sanitize_for_display

logger = logging.getLogger(__name__)


async def generate_streaming_response(
    request: Request,
    chat_model_wrapper: ChatModelWrapper,
    response_id: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_output_tokens: int,
    temperature: float,
    chat_tools: List[Dict[str, Any]] | None,
    tool_choice: str | None,
    instructions: str | None,
    previous_response_id: str | None,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response for the Responses API.

    Yields server-sent events (SSE) for:
    - response.created
    - reasoning.delta (if model produces thinking)
    - text.delta (for streaming text output)
    - output_item.added (for completed items)
    - response.done

    Args:
        request: FastAPI request object (for disconnect detection)
        chat_model_wrapper: Chat model instance
        response_id: Unique response ID
        model: Model name
        messages: Chat messages
        max_output_tokens: Max tokens to generate
        temperature: Sampling temperature
        chat_tools: Tool definitions (if any)
        tool_choice: Tool choice strategy
        instructions: System instructions
        previous_response_id: Previous response ID (for conversations)

    Yields:
        SSE strings (formatted as "event: <type>\ndata: <json>\n\n")
    """
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
    logger.debug(f"[STREAM] Sending response.created: {initial_response.to_dict()}")
    yield create_response_created_event(initial_response)

    # Track output for final response
    collected_content = []
    collected_thinking = []
    message_item_id = generate_message_id()
    reasoning_item_id = generate_reasoning_id()
    usage = ResponseUsage()
    in_tool_call = False  # Track if we're streaming a tool call
    json_brace_depth = 0  # Track JSON object depth for standalone JSON
    reasoning_sequence = 0  # Track sequence number for reasoning deltas

    # Log request for debugging (use DEBUG level to avoid console spam)
    logger.debug(f"[STREAM] Request to model: {len(messages)} messages")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")
        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
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

    logger.debug("[HANDLER] Starting to process stream chunks")

    # Helper to safely get next chunk (StopIteration -> sentinel)
    _sentinel = object()

    def safe_next(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return _sentinel

    # Iterate generator in threadpool to avoid blocking event loop
    stream_iter = iter(stream_response)
    try:
        while True:
            # Check for client disconnect before fetching next chunk
            if await request.is_disconnected():
                logger.debug("[HANDLER] Client disconnected, stopping stream")
                break

            # Get next chunk in threadpool (doesn't block event loop)
            try:
                chunk = await run_in_threadpool(safe_next, stream_iter)
            except (GeneratorExit, StopIteration):
                # Handle cases where the underlying generator is closed
                chunk = _sentinel

            if chunk is _sentinel:
                break

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                logger.debug(f"[HANDLER] Delta received: {delta}")

                # Handle reasoning/thinking (GPT-OSS Harmony -> OpenAI o-series)
                thinking = delta.get("thinking", "")
                if thinking:
                    logger.debug(f"[HANDLER] Thinking chunk: {repr(thinking[:100])}")
                    # Strip Harmony format directives from thinking text
                    cleaned_thinking = sanitize_for_display(thinking)
                    # Only emit non-empty thinking after sanitization
                    if cleaned_thinking:
                        collected_thinking.append(cleaned_thinking)
                        reasoning_sequence += 1
                        yield create_reasoning_delta_event(
                            response_id,
                            cleaned_thinking,
                            reasoning_item_id,
                            output_index=0,  # Reasoning is always first output
                            content_index=0,  # Single content block for reasoning
                            sequence_number=reasoning_sequence,
                        )

                # Handle text content
                content = delta.get("content", "")
                if content:
                    logger.debug(f"[HANDLER] Content chunk: {repr(content[:100])}")
                    # Always collect content for tool call parsing
                    collected_content.append(content)

                    # FIX: Don't stream tool call content as text
                    # Tool calls span multiple chunks, so we need state tracking

                    # Check if we're currently in a tool call BEFORE processing
                    was_in_tool_call = in_tool_call

                    # Detect start of Harmony format tool call
                    if "<|channel|>commentary" in content:
                        in_tool_call = True
                        logger.debug(
                            "[HANDLER] Detected Harmony tool call start "
                            "(commentary channel)"
                        )

                    # Detect end of Harmony format tool call
                    if "<|end|>" in content or "<|call|>" in content:
                        in_tool_call = False
                        logger.debug("[HANDLER] Detected Harmony tool call end")

                    # Detect standalone JSON tool calls
                    # Check for tool call patterns and track JSON depth
                    if not in_tool_call and (
                        '{"command"' in content
                        or '{"file_path"' in content
                        or '{"path"' in content
                        or '{"pattern"' in content
                        or '{"search_path"' in content
                        or '{"glob"' in content
                        or '{"old_string"' in content
                        or '{"notebook_path"' in content
                        or '{"url"' in content
                        or '{"query"' in content
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
                        # Sanitize content (preserve JSON - may be legitimate output)
                        clean_content = sanitize_for_display(content, strip_json=False)
                        if clean_content.strip():
                            logger.debug(
                                f"[HANDLER] Streaming content to user: "
                                f"{repr(clean_content[:100])}"
                            )
                            yield create_text_delta_event(
                                response_id, clean_content, message_item_id
                            )
                    else:
                        logger.debug(
                            f"[HANDLER] Suppressing tool call content: "
                            f"{repr(content[:50])}"
                        )

                # Note: Tool calls are NOT streamed by llama-cpp.
                # We parse them from collected_content after streaming.

                # Extract usage if present
                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    usage.input_tokens = chunk_usage.get("prompt_tokens", 0)
                    usage.output_tokens = chunk_usage.get("completion_tokens", 0)
                    usage.total_tokens = chunk_usage.get("total_tokens", 0)
    finally:
        # Crucial: Ensure the generator is closed to release the inference lock
        # in ChatModelWrapper, even if an exception occurs or client disconnects.
        if hasattr(stream_response, "close"):
            try:
                stream_response.close()
                logger.debug("[HANDLER] Stream generator closed in finally block")
            except Exception as e:
                logger.warning(f"[HANDLER] Error closing stream generator: {e}")

    # Build final response
    output = []
    # Keep raw content for tool call parsing, sanitize later for display
    raw_output_text = "".join(collected_content)
    output_index = 0

    # Include reasoning first (model thinks, then responds)
    # Use "summary" field (not "content") - Agent SDK expects this structure
    reasoning_text = "".join(collected_thinking)
    if reasoning_text:
        summary_item = {"type": "summary_text", "text": reasoning_text}
        reasoning_item = {
            "id": reasoning_item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [summary_item],  # Agent SDK expects "summary" not "content"
            "tool_calls": [],  # Empty for SDK compatibility
        }
        output.append(reasoning_item)
        # Emit output_item.added event for SDK
        logger.debug(f"[STREAM] Emitting reasoning item: {reasoning_item}")
        yield create_output_item_added_event(response_id, reasoning_item, output_index)
        # Emit reasoning.done event to signal reasoning completion
        yield create_reasoning_done_event(response_id, reasoning_item_id)
        output_index += 1

    # Parse tool calls from RAW content (before sanitization)
    # Tool call patterns need to be intact for Format 3 parsing
    parsed_tool_calls = []
    if chat_tools and raw_output_text:
        parsed_tool_calls = chat_model_wrapper._parse_tool_calls(raw_output_text) or []

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
                # Add empty content array for SDK compatibility
                "content": [],
            }
            output.append(fc_item)
            # Emit output_item.added event for SDK
            yield create_output_item_added_event(response_id, fc_item, output_index)
            output_index += 1
        # Clear text output when we have tool calls (like batch mode)
        output_text = ""
    else:
        # No tool calls - sanitize for display (preserve JSON in output)
        output_text = sanitize_for_display(raw_output_text, strip_json=False)

    # Always add message output (even if empty) to ensure SDK has a message item
    if output_text or not parsed_tool_calls:
        message_item = {
            "id": message_item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}]
            if output_text
            else [],
            "tool_calls": [],  # Empty for SDK compatibility
        }
        output.append(message_item)
        # Emit output_item.added event for SDK
        logger.debug(f"[STREAM] Emitting message item: {message_item}")
        yield create_output_item_added_event(response_id, message_item, output_index)
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
    logger.debug(f"[STREAM] Sending response.completed with {len(output)} items")
    logger.debug(f"[STREAM] Final response output: {output}")
    yield create_response_done_event(final_response)
