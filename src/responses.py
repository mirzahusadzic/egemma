"""
OpenAI Responses API Implementation

Implements the /v1/responses endpoint for OpenAI Agents SDK compatibility.
Maps to internal chat completions while providing the Responses API interface.

Key Features:
- Stateful conversations via previous_response_id
- Tool calling support
- Streaming via SSE
- Token usage tracking

Reference: https://platform.openai.com/docs/api-reference/responses
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ResponseInputItem:
    """Input item in a response request."""

    type: Literal["message", "item_reference"]
    role: str | None = None
    content: str | list[dict[str, Any]] | None = None
    item_id: str | None = None  # For item_reference type


@dataclass
class ResponseOutputMessage:
    """Message output item in a response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    content: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "status": self.status,
            "content": self.content,
        }


@dataclass
class ResponseOutputFunctionCall:
    """Function call output item in a response."""

    id: str
    call_id: str
    name: str
    arguments: str
    type: Literal["function_call"] = "function_call"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
        }


@dataclass
class ResponseOutputReasoning:
    """Reasoning/thinking output item (OpenAI o-series compatible).

    SDK reads "summary" field, converts to "content" internally.
    """

    id: str
    summary: list[dict[str, Any]] = field(default_factory=list)
    type: Literal["reasoning"] = "reasoning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "summary": self.summary,
        }


@dataclass
class ResponseUsage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        # OpenAI API uses snake_case for usage fields
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Response:
    """OpenAI Responses API response object."""

    id: str
    model: str
    output: list[dict[str, Any]] = field(default_factory=list)
    output_text: str = ""
    status: Literal["completed", "in_progress", "incomplete", "failed"] = "completed"
    object: Literal["response"] = "response"
    created_at: int = field(default_factory=lambda: int(time.time()))
    instructions: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    usage: ResponseUsage = field(default_factory=ResponseUsage)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "model": self.model,
            "output": self.output,
            "output_text": self.output_text,
            "status": self.status,
            "usage": self.usage.to_dict(),
        }

        if self.instructions is not None:
            result["instructions"] = self.instructions
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            result["max_output_tokens"] = self.max_output_tokens
        if self.previous_response_id is not None:
            result["previous_response_id"] = self.previous_response_id
        if self.error is not None:
            result["error"] = self.error

        return result


def generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"resp_{uuid.uuid4().hex[:24]}"


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def generate_call_id() -> str:
    """Generate a unique call ID for tool calls."""
    return f"call_{uuid.uuid4().hex[:24]}"


def generate_reasoning_id() -> str:
    """Generate a unique reasoning ID."""
    return f"rs_{uuid.uuid4().hex[:24]}"


def parse_input_to_messages(
    input_data: str | list[dict[str, Any]] | None,
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert Responses API input format to chat completions messages.

    Outputs Harmony-compatible format for GPT-OSS models:
    - Tool calls: {"name": "...", "arguments": "..."} (flat, not nested)
    - Tool results: {"role": "tool", "name": "...", "content": "..."}

    Args:
        input_data: String prompt or list of input items
        instructions: System instructions

    Returns:
        List of chat completion messages in Harmony format
    """
    messages = []

    # Add system message from instructions
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if input_data is None:
        return messages

    # Handle string input (simple prompt)
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    # Handle array of input items
    # Track pending tool calls to attach to assistant message
    # Harmony format: [{"name": "...", "arguments": "..."}]
    pending_tool_calls: list[dict[str, Any]] = []
    # Track call_id -> name mapping for tool results
    call_id_to_name: dict[str, str] = {}

    for item in input_data:
        item_type = item.get("type", "message")

        if item_type == "message":
            role = item.get("role", "user")
            content = item.get("content", "")

            # Handle content that might be a list of content parts
            if isinstance(content, list):
                # Convert to string for now (could support multimodal later)
                text_parts = [
                    p.get("text", "") for p in content if p.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            msg: dict[str, Any] = {"role": role, "content": content}

            # If there are pending tool calls and this is an assistant message,
            # attach them (they came from function_call items before this)
            if role == "assistant" and pending_tool_calls:
                msg["tool_calls"] = pending_tool_calls
                pending_tool_calls = []

            messages.append(msg)

        elif item_type == "function_call":
            # Tool call from assistant - use Harmony format (flat, not nested)
            call_id = item.get("call_id", item.get("id", ""))
            tool_name = item.get("name", "")

            # Track call_id -> name for later tool results
            if call_id and tool_name:
                call_id_to_name[call_id] = tool_name

            # Harmony format: flat structure without "function" wrapper
            tool_call = {
                "name": tool_name,
                "arguments": item.get("arguments", "{}"),
            }
            pending_tool_calls.append(tool_call)

        elif item_type == "function_call_output":
            # Tool call result - need assistant message with tool_calls before this
            # If we have pending tool calls, create assistant message first
            if pending_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": pending_tool_calls,
                    }
                )
                pending_tool_calls = []

            # Add tool response in Harmony format (uses "name" not "tool_call_id")
            call_id = item.get("call_id", "")
            tool_name = call_id_to_name.get(call_id, "unknown")
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": item.get("output", ""),
                }
            )

    # Handle any remaining pending tool calls
    if pending_tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": pending_tool_calls,
            }
        )

    return messages


def convert_tools_to_chat_format(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """
    Convert Responses API tool definitions to chat completions format.

    Args:
        tools: List of tool definitions in Responses API format

    Returns:
        List of tools in chat completions format, or None
    """
    if not tools:
        return None

    chat_tools = []
    for tool in tools:
        tool_type = tool.get("type", "function")

        if tool_type == "function":
            chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
            )

    return chat_tools if chat_tools else None


def convert_chat_response_to_response(
    chat_response: dict[str, Any],
    response_id: str,
    model: str,
    instructions: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    previous_response_id: str | None = None,
) -> Response:
    """
    Convert chat completions response to Responses API format.

    Args:
        chat_response: Response from chat completions
        response_id: ID for this response
        model: Model name
        instructions: Original instructions
        temperature: Original temperature
        max_output_tokens: Original max tokens
        previous_response_id: Previous response ID if continuing

    Returns:
        Response object
    """
    output = []
    output_text = ""

    # Extract message from chat response
    choices = chat_response.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        # Handle reasoning/thinking (GPT-OSS -> OpenAI o-series format)
        # Reasoning added FIRST (model thinks, then responds)
        # SDK reads "summary", converts to "content" internally
        thinking = message.get("thinking")
        if thinking:
            reasoning_output = ResponseOutputReasoning(
                id=generate_reasoning_id(),
                summary=[{"type": "summary_text", "text": thinking}],
            )
            output.append(reasoning_output.to_dict())

        # Handle text content
        content = message.get("content", "")
        if content:
            output_text = content
            msg_output = ResponseOutputMessage(
                id=generate_message_id(),
                content=[{"type": "output_text", "text": content}],
            )
            output.append(msg_output.to_dict())

        # Handle tool calls
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            fc_output = ResponseOutputFunctionCall(
                id=generate_message_id(),
                call_id=tc.get("id", generate_call_id()),
                name=func.get("name", ""),
                arguments=func.get("arguments", "{}"),
            )
            output.append(fc_output.to_dict())

    # Extract usage
    usage_data = chat_response.get("usage", {})
    usage = ResponseUsage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )

    return Response(
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


def create_stream_event(event_type: str, data: dict[str, Any], response_id: str) -> str:
    """
    Create a Server-Sent Event for streaming.

    Args:
        event_type: Type of event (e.g., "response.created")
        data: Event data
        response_id: Response ID

    Returns:
        SSE-formatted string
    """
    event_data = {"type": event_type, "response_id": response_id, **data}
    return f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"


def create_response_created_event(response: Response) -> str:
    """Create response.created streaming event."""
    return create_stream_event(
        "response.created",
        {"response": response.to_dict()},
        response.id,
    )


def create_text_delta_event(response_id: str, delta: str, item_id: str) -> str:
    """Create response.output_text.delta streaming event."""
    return create_stream_event(
        "response.output_text.delta",
        {"delta": delta, "item_id": item_id},
        response_id,
    )


def create_reasoning_delta_event(response_id: str, delta: str, item_id: str) -> str:
    """Create reasoning_summary.delta streaming event for thinking."""
    return create_stream_event(
        "response.reasoning_summary.delta",
        {"delta": delta, "item_id": item_id},
        response_id,
    )


def create_output_item_added_event(
    response_id: str, item: dict[str, Any], output_index: int
) -> str:
    """
    Create response.output_item.added streaming event.

    This event notifies the SDK that a new output item has been added.
    Required for the OpenAI Agents SDK to properly track output items.

    Args:
        response_id: ID of the response
        item: The output item dict (message, function_call, reasoning)
        output_index: Index of the item in the output array

    Returns:
        SSE-formatted string
    """
    return create_stream_event(
        "response.output_item.added",
        {"item": item, "output_index": output_index},
        response_id,
    )


def create_response_done_event(response: Response) -> str:
    """Create response.completed streaming event."""
    return create_stream_event(
        "response.completed",
        {"response": response.to_dict()},
        response.id,
    )
