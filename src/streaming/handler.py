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
    create_response_created_event,
    create_response_done_event,
    create_text_delta_event,
    generate_call_id,
    generate_message_id,
    generate_reasoning_id,
)
from ..models.llm import ChatModelWrapper
from ..util import sanitize_thinking

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
                # Strip Harmony format directives from thinking text
                cleaned_thinking = sanitize_thinking(thinking)
                # Only emit non-empty thinking after sanitization
                if cleaned_thinking:
                    collected_thinking.append(cleaned_thinking)
                    yield create_reasoning_delta_event(
                        response_id, cleaned_thinking, reasoning_item_id
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
                    yield create_text_delta_event(response_id, content, message_item_id)

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
        yield create_output_item_added_event(response_id, reasoning_item, output_index)
        output_index += 1

    # Parse tool calls from collected content (streaming doesn't
    # emit tool_calls, so we parse from text like batch mode)
    parsed_tool_calls = []
    if chat_tools and output_text:
        parsed_tool_calls = chat_model_wrapper._parse_tool_calls(output_text) or []

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
            yield create_output_item_added_event(response_id, fc_item, output_index)
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
    yield create_response_done_event(final_response)
