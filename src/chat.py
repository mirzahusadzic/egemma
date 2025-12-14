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
            # chat_format auto-detects from GGUF; Harmony parsing handled manually
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
        """Format messages with tool definitions and Harmony settings."""
        # Build Harmony system prompt additions
        harmony_header = self._build_harmony_header()
        tool_prompt = self._build_tool_prompt(tools) if tools else ""

        # Combine into suffix for system message
        suffix_parts = [p for p in [harmony_header, tool_prompt] if p]
        suffix = "\n\n".join(suffix_parts)

        if not suffix:
            return messages

        formatted = []
        has_system = False

        for msg in messages:
            if msg["role"] == "system":
                # Append Harmony settings and tool definitions to system message
                formatted.append(
                    {"role": "system", "content": f"{msg['content']}\n\n{suffix}"}
                )
                has_system = True
            else:
                formatted.append(msg)

        # If no system message, add one with Harmony settings
        if not has_system:
            formatted.insert(0, {"role": "system", "content": suffix})

        return formatted

    def _build_harmony_header(self) -> str:
        """Build Harmony format header with reasoning effort."""
        from datetime import datetime

        current_date = datetime.now().strftime("%Y-%m-%d")
        reasoning_effort = settings.CHAT_REASONING_EFFORT

        return f"Reasoning: {reasoning_effort}\nCurrent date: {current_date}"

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
        return "You have access to the following tools:\n\n" + "\n\n".join(
            tool_descriptions
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
        # Pass tools to llama.cpp so it can apply the Harmony template automatically
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,  # Let llama.cpp apply the chat template with tools
            stop=None,
            stream=False,  # Batch mode, not streaming
            min_p=0.05,  # Prevents degenerate sampling
            top_p=1.0,  # Unsloth recommendation
            top_k=40,  # Adds diversity (consistent across both modes)
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

        # Get token counts from response
        completion_tokens = response.get("usage", {}).get("completion_tokens", 0)

        # If llama-cpp didn't provide token counts, estimate them
        if completion_tokens == 0 and content:
            completion_tokens = self._count_tokens(content)

        # Determine finish_reason
        # - tool_calls: model requested tool execution
        # - length: max_tokens was reached
        # - stop: normal completion
        if tool_calls:
            finish_reason = "tool_calls"
        elif completion_tokens >= max_tokens - 10:  # Within 10 tokens of limit
            finish_reason = "length"
        else:
            finish_reason = "stop"

        prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        if prompt_tokens == 0:
            # Estimate prompt tokens if not provided
            prompt_text = "".join(str(msg.get("content", "")) for msg in messages)
            prompt_tokens = self._count_tokens(prompt_text)

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
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
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

        OpenAI SDK Compatibility:
        - Tracks token counts during streaming
        - Includes `usage` object in the final chunk (required by @openai/agents SDK)
        - Detects finish_reason="length" when max_tokens is reached
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Estimate prompt tokens from input messages
        prompt_text = "".join(str(msg.get("content", "")) for msg in messages)
        prompt_tokens = self._count_tokens(prompt_text)

        # Pass tools to llama.cpp so it can apply the Harmony template automatically
        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,  # Let llama.cpp apply the chat template with tools
            stop=None,
            stream=True,
            min_p=0.05,  # Prevents degenerate sampling
            top_p=1.0,  # Unsloth recommendation
            top_k=40,  # Adds diversity (consistent across both modes)
        )

        # Track Harmony format state
        harmony_detected = False
        in_thinking = False
        in_final = False
        thinking_emitted = False
        final_content_streamed = False
        tool_prefix_emitted = False  # Track if we've emitted tool prefix
        buffer = ""

        # Track completion tokens for final usage report
        completion_tokens = 0
        total_content_chars = 0

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
                    # Check for full analysis marker including <|start|>assistant
                    # prefix (per template line 292)
                    if (
                        "<|start|>assistant<|channel|>analysis<|message|>" in buffer
                        or "<|channel|>analysis<|message|>" in buffer
                    ) and not in_thinking:
                        in_thinking = True
                        # Handle both full and partial markers for robustness
                        if "<|start|>assistant<|channel|>analysis<|message|>" in buffer:
                            buffer = buffer.split(
                                "<|start|>assistant<|channel|>analysis<|message|>"
                            )[-1]
                        else:
                            buffer = buffer.split("<|channel|>analysis<|message|>")[-1]

                    # Check for final channel OR commentary channel (tool calls)
                    # Tool calls: <|channel|>commentary to=functions.X ...
                    # Only detect commentary transition if we're not already
                    # in final state
                    elif not in_final and (
                        "<|channel|>final<|message|>" in buffer
                        or (
                            "<|channel|>commentary" in buffer
                            and "<|message|>" in buffer
                        )
                    ):
                        # Extract tool call prefix for commentary channel
                        # IMPORTANT: Preserve full Harmony format for parser
                        # Template line 297:
                        # to=functions.{name}<|channel|>commentary json<|message|>{...}
                        tool_prefix = ""
                        if "<|channel|>commentary" in buffer:
                            import re

                            # Extract the FULL tool call prefix
                            # (preserve all markers!)
                            # NOTE: Model outputs REVERSED order from template!
                            # Model: <|channel|>commentary to={name}
                            #        <|constrain|>json<|message|>
                            # Template: to=functions.{name}<|channel|>commentary
                            #           json<|message|>
                            match = re.search(
                                r"(<\|channel\|>commentary\s+to=\s*[\w._]+\s*(?:<\|constrain\|>)?(?:json|code)?\s*<\|message\|>)",
                                buffer,
                            )
                            if match:
                                tool_prefix = match.group(1)

                        # Determine which channel marker to split on
                        if "<|channel|>final<|message|>" in buffer:
                            split_marker = "<|channel|>final<|message|>"
                        else:
                            # For commentary, split on <|message|> after it
                            split_marker = "<|message|>"

                        # Emit thinking before transitioning to final
                        if in_thinking and include_thinking and not thinking_emitted:
                            thinking_content = buffer.split(split_marker)[0]
                            # Clean up to just get the thinking part
                            if "<|channel|>commentary" in thinking_content:
                                thinking_content = thinking_content.split(
                                    "<|channel|>commentary"
                                )[0]
                            thinking_content = (
                                thinking_content.replace("<|end|>", "")
                                .replace("<|start|>", "")
                                .replace("<|channel|>", "")
                                .replace("<|call|>", "")
                                .replace("<|message|>", "")
                                .replace("assistant", "")
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
                        # Extract JSON part that comes after the tool prefix
                        json_part = buffer.split(split_marker)[-1]
                        # Prepend tool prefix for proper parsing later
                        buffer = tool_prefix + json_part

                        # Emit tool prefix immediately so it's included in output (ONCE)
                        if tool_prefix and not tool_prefix_emitted:
                            final_content_streamed = True
                            tool_prefix_emitted = True  # Mark as emitted
                            total_content_chars += len(tool_prefix)
                            yield {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": tool_prefix},
                                        "finish_reason": None,
                                    }
                                ],
                            }

                            # Also emit the JSON part that's already in the buffer
                            if json_part:
                                total_content_chars += len(json_part)
                                yield {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": json_part},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                # Clear the buffer since we've emitted everything
                                buffer = ""

                    elif in_final:
                        # Stream final content, cleaning Harmony tokens
                        clean = content
                        harmony_tokens = [
                            "<|end|>",
                            "<|start|>",
                            "<|channel|>",
                            "<|call|>",
                            "<|message|>",
                        ]
                        for tok in harmony_tokens:
                            clean = clean.replace(tok, "")
                        if clean:
                            final_content_streamed = True
                            total_content_chars += len(clean)
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
                    total_content_chars += len(content)
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
                        total_content_chars += len(parsed["content"])
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

                # Emit any remaining buffer content when in_final mode
                # This handles tool calls that arrived in the same chunk as
                # <|channel|>final<|message|> and weren't streamed yet
                # Skip if we already streamed final content incrementally
                if (
                    harmony_detected
                    and in_final
                    and buffer
                    and not final_content_streamed
                ):
                    # Clean Harmony tokens from buffer
                    clean_buffer = buffer
                    harmony_toks = [
                        "<|end|>",
                        "<|start|>",
                        "<|channel|>",
                        "<|call|>",
                        "<|message|>",
                    ]
                    for tok in harmony_toks:
                        clean_buffer = clean_buffer.replace(tok, "")
                    clean_buffer = clean_buffer.strip()
                    if clean_buffer:
                        total_content_chars += len(clean_buffer)
                        yield {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": clean_buffer},
                                    "finish_reason": None,
                                }
                            ],
                        }

                # Calculate completion tokens from content
                completion_tokens = self._count_tokens_from_chars(total_content_chars)

                # Detect finish_reason="length" when max_tokens is reached
                # llama-cpp often returns "stop" even when hitting max_tokens
                final_finish_reason = finish_reason
                if completion_tokens >= max_tokens - 10:  # Within 10 tokens of limit
                    final_finish_reason = "length"

                # Yield final chunk with usage (OpenAI SDK requirement)
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": final_finish_reason}
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }

    def _parse_tool_calls(self, content: str) -> list[dict] | None:
        """
        Parse tool calls from model output.

        Supports two formats:
        1. OpenAI JSON format: {"tool_calls": [...]}
        2. GPT-OSS Harmony format: to=tool <name> json{...}

        Returns OpenAI-compatible tool_calls array.
        """
        import re

        # Format 1: Try to find JSON with tool_calls
        try:
            if '"tool_calls"' in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                    if "tool_calls" in data:
                        return data["tool_calls"]
        except json.JSONDecodeError:
            pass

        # Format 2: Parse GPT-OSS Harmony tool format
        # NOTE: Model OUTPUT format differs from template INPUT format!
        # Template (input): to=functions.{name}<|channel|>commentary
        #                   json<|message|>{args}
        # Model (output):   <|channel|>commentary to={name}
        #                   <|constrain|>json<|message|>{args}
        #
        # Tool names can contain dots (e.g., container.exec) and underscores.
        # Strip common prefixes like "functions." or "tool." from the name.
        tool_pattern = (
            # Model outputs commentary BEFORE to=
            r"<\|channel\|>commentary\s+to=\s*([\w._]+)"
            # Optional <|constrain|>
            r"\s*(?:<\|constrain\|>)?(?:json|code)?\s*<\|message\|>\s*\{"
        )

        tool_calls = []
        matches = list(re.finditer(tool_pattern, content, re.IGNORECASE))
        for match in matches:
            tool_name = match.group(1)
            # Strip common prefixes (model outputs "to=functions.X" or "to=tool.X")
            if tool_name.startswith("functions."):
                tool_name = tool_name[10:]  # len("functions.") == 10
            elif tool_name.startswith("tool."):
                tool_name = tool_name[5:]  # len("tool.") == 5
            json_start = match.end() - 1  # Position of opening brace

            # Extract balanced JSON by counting braces
            args_json = self._extract_balanced_json(content, json_start)
            if args_json:
                try:
                    parsed = json.loads(args_json)
                    # Handle nested format: {"id":..., "function":{...}}
                    if isinstance(parsed, dict) and "function" in parsed:
                        nested_func = parsed["function"]
                        actual_name = nested_func.get("name", tool_name)
                        actual_args = nested_func.get("arguments", "{}")
                        # Strip prefixes from nested name too
                        if actual_name.startswith("functions."):
                            actual_name = actual_name[10:]
                        elif actual_name.startswith("tool."):
                            actual_name = actual_name[5:]
                        tool_calls.append(
                            {
                                "id": parsed.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": actual_name,
                                    "arguments": (
                                        actual_args
                                        if isinstance(actual_args, str)
                                        else json.dumps(actual_args)
                                    ),
                                },
                            }
                        )
                    else:
                        # Simple format: args_json is the actual arguments
                        tool_calls.append(
                            {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": args_json,
                                },
                            }
                        )
                except json.JSONDecodeError:
                    # Attempt to repair common JSON errors from model output
                    if args_json:
                        repaired_json = self._repair_json(args_json)
                        if repaired_json != args_json:
                            try:
                                parsed = json.loads(repaired_json)
                                # Successfully repaired! Process the tool call
                                if isinstance(parsed, dict) and "function" in parsed:
                                    nested_func = parsed["function"]
                                    actual_name = nested_func.get("name", tool_name)
                                    actual_args = nested_func.get("arguments", "{}")
                                    if actual_name.startswith("functions."):
                                        actual_name = actual_name[10:]
                                    elif actual_name.startswith("tool."):
                                        actual_name = actual_name[5:]
                                    tool_calls.append(
                                        {
                                            "id": parsed.get(
                                                "id", f"call_{uuid.uuid4().hex[:8]}"
                                            ),
                                            "type": "function",
                                            "function": {
                                                "name": actual_name,
                                                "arguments": (
                                                    actual_args
                                                    if isinstance(actual_args, str)
                                                    else json.dumps(actual_args)
                                                ),
                                            },
                                        }
                                    )
                                else:
                                    # Simple format
                                    tool_calls.append(
                                        {
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": repaired_json,
                                            },
                                        }
                                    )
                                continue
                            except json.JSONDecodeError:
                                pass
                    continue

        # Format 3: Detect standalone JSON that looks like a tool call
        # Example: {"command":"ls -R","timeout":120}
        # This happens when the model outputs raw JSON without markers
        if not tool_calls:
            try:
                # Try to find a JSON object in the content
                start = content.find("{")
                if start >= 0:
                    args_json = self._extract_balanced_json(content, start)
                    if args_json:
                        parsed = json.loads(args_json)
                        if isinstance(parsed, dict):
                            # Detect tool type from structure
                            tool_name = None
                            if "command" in parsed:
                                tool_name = "bash"
                            elif "function" in parsed or "tool" in parsed:
                                # Could be a nested tool call format
                                tool_name = parsed.get(
                                    "function", parsed.get("tool", None)
                                )

                            if tool_name:
                                tool_calls.append(
                                    {
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": args_json,
                                        },
                                    }
                                )
            except (json.JSONDecodeError, ValueError):
                pass

        return tool_calls if tool_calls else None

    def _extract_balanced_json(self, content: str, start: int) -> str | None:
        """Extract a balanced JSON object starting at the given position."""
        if start >= len(content) or content[start] != "{":
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(content)):
            char = content[i]

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]

        return None

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON errors from model output.

        Common errors:
        1. Mixed brackets/quotes: "] instead of "
        2. Trailing commas before } or ]

        Returns repaired JSON string (or original if no repairs needed).
        """
        import re

        # Fix 1: Replace "] with " (bracket/quote confusion)
        # Example: {"command":"git status -s"] -> {"command":"git status -s"
        json_str = re.sub(r'"\]', '"', json_str)

        # Fix 2: Remove trailing commas before } or ]
        # Example: {"a": 1,} -> {"a": 1}
        # Example: ["a", "b",] -> ["a", "b"]
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

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

        logger.debug(f"[PARSE_HARMONY] Raw content: {content[:500]}")

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

        # Extract commentary channel for tool calls
        # NOTE: Model output format:
        # <|channel|>commentary to={name} <|constrain|>json<|message|>{...}
        # IMPORTANT: Preserve full Harmony format for _parse_tool_calls
        if not final_match or not final_content:
            commentary_match = re.search(
                r"(<\|channel\|>commentary\s+to=\s*[\w._]+\s*(?:<\|constrain\|>)?(?:json|code)?\s*<\|message\|>\s*\{.*?)(?:<\|end\|>|<\|call\|>|$)",
                content,
                re.DOTALL,
            )
            if commentary_match:
                # Preserve the FULL Harmony format including all markers
                final_content = commentary_match.group(1).strip()

        if not final_content and thinking:
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

        logger.debug(
            f"[PARSE_HARMONY] Extracted - thinking: {bool(thinking)}, "
            f"content: {final_content[:200] if final_content else 'None'}"
        )
        return {"thinking": thinking, "content": final_content or content}

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.

        Uses llama-cpp-python's tokenizer for accurate token counts.
        Falls back to character-based estimation if model not loaded.
        """
        if self.model is None:
            return self._count_tokens_from_chars(len(text))

        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception:
            # Fallback to estimation
            return self._count_tokens_from_chars(len(text))

    def _count_tokens_from_chars(self, char_count: int) -> int:
        """
        Estimate token count from character count.

        Uses a ratio of ~4 characters per token (common for English text).
        This is a rough estimate when tokenizer is not available.
        """
        return max(1, char_count // 4)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
