"""
OpenAI API Compatibility Layer

Pydantic models matching the OpenAI Chat Completions API format.
Used for request/response validation and serialization.
"""

from typing import Literal

from pydantic import BaseModel, Field

# --- Request Models ---


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""

    type: str = "object"
    properties: dict = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: str | None = None
    parameters: FunctionParameters | dict = Field(default_factory=dict)


class Tool(BaseModel):
    """Tool definition (currently only function type)."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatMessage(BaseModel):
    """A message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    user: str | None = None
    # Extended thinking support (GPT-OSS Harmony format)
    include_thinking: bool = Field(
        default=False,
        description="Include model's internal reasoning/analysis in response",
    )


# --- Response Models ---


class FunctionCall(BaseModel):
    """Function call in assistant message."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call in assistant response."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    """Message in completion response."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    # Extended thinking (GPT-OSS Harmony analysis channel)
    thinking: str | None = None


class Choice(BaseModel):
    """A completion choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# --- Streaming Response Models ---


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None
    # Extended thinking (GPT-OSS Harmony analysis channel)
    thinking: str | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    # Usage is included in the final chunk (OpenAI SDK requirement)
    usage: Usage | None = None


# --- Models Endpoint ---


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "egemma"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]
