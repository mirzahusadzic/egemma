"""
Chat Model Wrapper for GPT-OSS-20B

Provides OpenAI-compatible chat completions using llama-cpp-python.
Supports streaming and tool calling for agentic workflows.
"""

import json
import logging
import time
import uuid
from typing import Iterator

from llama_cpp import Llama

from src.config import settings

logger = logging.getLogger(__name__)


class ChatModelWrapper:
    """Wrapper for GPT-OSS-20B chat model using llama-cpp-python."""

    def __init__(self):
        self.model: Llama | None = None
        self.model_name = settings.CHAT_MODEL_NAME

    def load_model(self) -> None:
        """Load the GGUF model with Metal acceleration."""
        logger.info(f"Loading chat model from {settings.CHAT_MODEL_PATH}...")

        self.model = Llama(
            model_path=settings.CHAT_MODEL_PATH,
            n_ctx=settings.CHAT_N_CTX,
            n_gpu_layers=settings.CHAT_N_GPU_LAYERS,
            n_batch=settings.CHAT_N_BATCH,
            use_mmap=settings.CHAT_USE_MMAP,
            flash_attn=settings.CHAT_FLASH_ATTN,
            verbose=False,
            # GPT-OSS uses Harmony format - let model's Jinja template handle it
        )

        logger.info(
            f"Chat model loaded: {settings.CHAT_MODEL_NAME} "
            f"(ctx={settings.CHAT_N_CTX}, gpu_layers={settings.CHAT_N_GPU_LAYERS}, "
            f"batch={settings.CHAT_N_BATCH}, flash_attn={settings.CHAT_FLASH_ATTN})"
        )

    def create_completion(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        include_thinking: bool = False,
    ) -> dict | Iterator[dict]:
        """
        Create a chat completion (OpenAI-compatible).

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: List of tool definitions (OpenAI function calling format)
            tool_choice: Tool selection mode ("auto", "none", or specific tool)
            stream: Whether to stream the response
            include_thinking: Whether to include GPT-OSS thinking/analysis blocks

        Returns:
            OpenAI-compatible completion response or stream iterator
        """
        if self.model is None:
            raise RuntimeError("Chat model not loaded")

        max_tokens = max_tokens or settings.CHAT_MAX_TOKENS
        temperature = (
            temperature if temperature is not None else settings.CHAT_TEMPERATURE
        )

        # Build the prompt with tool definitions if provided
        formatted_messages = self._format_messages_with_tools(messages, tools)

        if stream:
            return self._stream_completion(
                formatted_messages, max_tokens, temperature, tools, include_thinking
            )
        else:
            return self._batch_completion(
                formatted_messages, max_tokens, temperature, tools, include_thinking
            )

    def _format_messages_with_tools(
        self, messages: list[dict], tools: list[dict] | None
    ) -> list[dict]:
        """Format messages with tool definitions in system prompt."""
        if not tools:
            return messages

        # Add tool definitions to system message
        tool_prompt = self._build_tool_prompt(tools)

        formatted = []
        has_system = False

        for msg in messages:
            if msg["role"] == "system":
                # Append tool definitions to existing system message
                formatted.append(
                    {"role": "system", "content": f"{msg['content']}\n\n{tool_prompt}"}
                )
                has_system = True
            else:
                formatted.append(msg)

        # If no system message, add one with tool definitions
        if not has_system:
            formatted.insert(0, {"role": "system", "content": tool_prompt})

        return formatted

    def _build_tool_prompt(self, tools: list[dict]) -> str:
        """Build tool description prompt for the model."""
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                tool_descriptions.append(
                    f"- {func['name']}: {func.get('description', '')}\n"
                    f"  Parameters: {json.dumps(func.get('parameters', {}))}"
                )

        return (
            "You have access to the following tools:\n\n"
            + "\n\n".join(tool_descriptions)
            + "\n\nTo use a tool, respond with a JSON object in this format:\n"
            '{"tool_calls": [{"id": "call_xxx", "type": "function", '
            '"function": {"name": "tool_name", "arguments": "{...}"}}]}\n\n'
            "Only use tools when necessary. Respond normally for regular conversation."
        )

    def _batch_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        include_thinking: bool = False,
    ) -> dict:
        """Generate a non-streaming completion."""
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # Don't stop on Harmony tokens - we need to parse them
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        # Parse Harmony format response
        raw_content = response["choices"][0]["message"]["content"]
        parsed = self._parse_harmony_response(raw_content)
        content = parsed["content"]
        thinking = parsed["thinking"]

        # Parse tool calls from response if tools were provided
        tool_calls = None
        if tools and content:
            tool_calls = self._parse_tool_calls(content)
            if tool_calls:
                # Clear content if we have tool calls
                content = None

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        result = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0),
            },
        }

        # Include thinking blocks if requested (Cognition CLI extended thinking)
        if include_thinking and thinking:
            result["choices"][0]["message"]["thinking"] = thinking

        return result

    def _stream_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        include_thinking: bool = False,
    ) -> Iterator[dict]:
        """
        Generate a streaming completion with Harmony format parsing.

        When include_thinking=True, emits thinking chunks first, then content chunks.
        Uses a buffered approach to properly parse Harmony format tokens.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # Don't stop on Harmony tokens - we need to parse them
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )

        # Track Harmony format state
        harmony_detected = False
        in_thinking = False
        in_final = False
        thinking_emitted = False
        buffer = ""

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            finish_reason = chunk["choices"][0].get("finish_reason")
            content = delta.get("content", "")

            if content:
                buffer += content

                # Check for Harmony format markers
                if "<|channel|>" in buffer:
                    harmony_detected = True

                if harmony_detected:
                    # Harmony format parsing
                    if "<|channel|>analysis<|message|>" in buffer and not in_thinking:
                        in_thinking = True
                        buffer = buffer.split("<|channel|>analysis<|message|>")[-1]

                    elif "<|channel|>final<|message|>" in buffer:
                        # Emit thinking before transitioning to final
                        if in_thinking and include_thinking and not thinking_emitted:
                            thinking_content = buffer.split(
                                "<|channel|>final<|message|>"
                            )[0]
                            thinking_content = (
                                thinking_content.replace("<|end|>", "")
                                .replace("<|start|>", "")
                                .strip()
                            )
                            if thinking_content:
                                yield {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"thinking": thinking_content},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            thinking_emitted = True

                        in_thinking = False
                        in_final = True
                        buffer = buffer.split("<|channel|>final<|message|>")[-1]

                    elif in_final:
                        # Stream final content, cleaning Harmony tokens
                        clean = content
                        harmony_tokens = ["<|end|>", "<|start|>", "<|channel|>"]
                        for tok in harmony_tokens + ["<|message|>"]:
                            clean = clean.replace(tok, "")
                        if clean:
                            yield {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": clean},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                else:
                    # Non-Harmony: pass through as-is (OpenAI compatible)
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }

            if finish_reason:
                # For Harmony format that didn't complete cleanly, emit buffered
                if harmony_detected and in_thinking and not thinking_emitted:
                    parsed = self._parse_harmony_response(buffer)
                    if include_thinking and parsed["thinking"]:
                        yield {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"thinking": parsed["thinking"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    if parsed["content"] and parsed["content"] != buffer:
                        yield {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": parsed["content"]},
                                    "finish_reason": None,
                                }
                            ],
                        }

                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }

    def _parse_tool_calls(self, content: str) -> list[dict] | None:
        """Parse tool calls from model output."""
        try:
            # Try to find JSON with tool_calls
            if '"tool_calls"' in content:
                # Extract JSON from content
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                    if "tool_calls" in data:
                        return data["tool_calls"]
        except json.JSONDecodeError:
            pass
        return None

    def _parse_harmony_response(self, content: str) -> dict:
        """
        Parse Harmony format response into thinking + final answer.

        Harmony format:
            <|channel|>analysis<|message|> ...thinking... <|end|>
            <|start|>assistant<|channel|>final<|message|> ...answer... <|end|>

        Returns:
            {"thinking": str | None, "content": str}
        """
        import re

        thinking = None
        final_content = content  # Fallback to raw content

        # Extract analysis/thinking channel
        analysis_match = re.search(
            r"<\|channel\|>analysis<\|message\|>\s*(.*?)(?:<\|end\|>|<\|start\|>)",
            content,
            re.DOTALL,
        )
        if analysis_match:
            thinking = analysis_match.group(1).strip()

        # Extract final channel (the actual response)
        final_match = re.search(
            r"<\|channel\|>final<\|message\|>\s*(.*?)(?:<\|end\|>|$)",
            content,
            re.DOTALL,
        )
        if final_match:
            final_content = final_match.group(1).strip()
        elif thinking:
            # If we found thinking but no final, the whole non-thinking part is content
            # This handles edge cases where model doesn't use final channel
            final_content = re.sub(
                r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|<\|start\|>)",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            # Clean any remaining Harmony tokens
            final_content = re.sub(r"<\|[^|]+\|>", "", final_content).strip()

        return {"thinking": thinking, "content": final_content or content}

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
