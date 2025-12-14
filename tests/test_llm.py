"""Tests for ChatModelWrapper - GPT-OSS-20B chat completions."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.llm import ChatModelWrapper


def test_chat_wrapper_init():
    """Test ChatModelWrapper initialization."""
    wrapper = ChatModelWrapper()
    assert wrapper.model is None
    assert wrapper.model_name == "gpt-oss-20b"


def test_is_loaded_false():
    """Test is_loaded property when model not loaded."""
    wrapper = ChatModelWrapper()
    assert wrapper.is_loaded is False


def test_is_loaded_true():
    """Test is_loaded property when model is loaded."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()
    assert wrapper.is_loaded is True


def test_load_model():
    """Test loading the GGUF model with M3-optimized settings."""
    wrapper = ChatModelWrapper()

    with patch("src.models.llm.Llama") as mock_llama:
        mock_model = MagicMock()
        mock_llama.return_value = mock_model

        wrapper.load_model()

        assert wrapper.model is not None
        mock_llama.assert_called_once()

        # Verify M3 Pro optimizations are passed
        call_kwargs = mock_llama.call_args.kwargs
        assert "n_batch" in call_kwargs
        assert "use_mmap" in call_kwargs
        assert "flash_attn" in call_kwargs
        assert call_kwargs["use_mmap"] is False  # Prevents M3 freeze
        assert call_kwargs["flash_attn"] is True  # Metal optimization


def test_create_completion_model_not_loaded():
    """Test create_completion raises error when model not loaded."""
    wrapper = ChatModelWrapper()

    with pytest.raises(RuntimeError, match="Chat model not loaded"):
        wrapper.create_completion(messages=[{"role": "user", "content": "Hi"}])


def test_batch_completion():
    """Test non-streaming completion."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()
    wrapper.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
        temperature=0.7,
        stream=False,
    )

    assert result["object"] == "chat.completion"
    assert result["model"] == "gpt-oss-20b"
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 15


def test_batch_completion_with_tool_calls():
    """Test completion that includes tool calls."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()

    tool_response = (
        '{"tool_calls": [{"id": "call_1", "type": "function", '
        '"function": {"name": "get_weather", "arguments": "{}"}}]}'
    )
    wrapper.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": tool_response}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        stream=False,
    )

    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["choices"][0]["message"]["tool_calls"] is not None
    assert result["choices"][0]["message"]["content"] is None


def test_stream_completion():
    """Test streaming completion."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()

    # Mock streaming response
    mock_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    wrapper.model.create_chat_completion.return_value = iter(mock_chunks)

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = list(result)
    assert len(chunks) == 3
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


def test_format_messages_with_tools_no_tools():
    """Test message formatting without tools adds Harmony header."""
    wrapper = ChatModelWrapper()
    messages = [{"role": "user", "content": "Hello"}]

    result = wrapper._format_messages_with_tools(messages, None)

    # Harmony header is always added (reasoning effort + date)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "Reasoning:" in result[0]["content"]
    assert result[1] == messages[0]


def test_format_messages_with_tools_existing_system():
    """Test message formatting with tools and existing system message."""
    wrapper = ChatModelWrapper()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "test", "description": "Test tool", "parameters": {}},
        }
    ]

    result = wrapper._format_messages_with_tools(messages, tools)

    assert len(result) == 2
    assert "You are helpful." in result[0]["content"]
    assert "test" in result[0]["content"]


def test_format_messages_with_tools_no_system():
    """Test message formatting with tools and no system message."""
    wrapper = ChatModelWrapper()
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "test", "description": "Test tool", "parameters": {}},
        }
    ]

    result = wrapper._format_messages_with_tools(messages, tools)

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "test" in result[0]["content"]


def test_build_tool_prompt():
    """Test building tool prompt from definitions."""
    wrapper = ChatModelWrapper()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {},
            },
        },
    ]

    result = wrapper._build_tool_prompt(tools)

    assert "get_weather" in result
    assert "Get current weather" in result
    assert "search" in result
    assert "Search the web" in result
    assert "tools" in result  # Just lists available tools


def test_parse_tool_calls_valid():
    """Test parsing valid tool calls from content."""
    wrapper = ChatModelWrapper()
    content = (
        '{"tool_calls": [{"id": "call_1", "type": "function", '
        '"function": {"name": "test", "arguments": "{}"}}]}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "call_1"


def test_parse_tool_calls_no_tool_calls():
    """Test parsing content without tool calls."""
    wrapper = ChatModelWrapper()
    content = "Just a regular response without tools."

    result = wrapper._parse_tool_calls(content)

    assert result is None


def test_parse_tool_calls_invalid_json():
    """Test parsing invalid JSON."""
    wrapper = ChatModelWrapper()
    content = '{"tool_calls": invalid json here'

    result = wrapper._parse_tool_calls(content)

    assert result is None


def test_parse_tool_calls_embedded_json():
    """Test parsing tool calls embedded in text."""
    wrapper = ChatModelWrapper()
    content = (
        'Some text before {"tool_calls": [{"id": "call_1", '
        '"type": "function", "function": {"name": "test", '
        '"arguments": "{}"}}]} and after'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert result[0]["id"] == "call_1"


def test_completion_default_values():
    """Test completion uses default values from settings."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()
    wrapper.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hi"}}],
        "usage": {},
    }

    with patch("src.models.llm.settings") as mock_settings:
        mock_settings.CHAT_MAX_TOKENS = 2048
        mock_settings.CHAT_TEMPERATURE = 0.5

        wrapper.create_completion(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
        )

        call_args = wrapper.model.create_chat_completion.call_args
        assert call_args.kwargs["max_tokens"] == 2048
        assert call_args.kwargs["temperature"] == 0.5


def test_stream_completion_with_harmony_format():
    """Test streaming with Harmony format (analysis + final channels)."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()

    # Mock Harmony format streaming response
    mock_chunks = [
        {"choices": [{"delta": {"content": "<|channel|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "analysis"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|message|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Thinking..."}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|end|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|channel|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "final"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|message|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Hello!"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    wrapper.model.create_chat_completion.return_value = iter(mock_chunks)

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        include_thinking=True,
    )

    chunks = list(result)
    # Should have thinking chunk, content chunk, and final chunk
    assert len(chunks) >= 2

    # Find thinking and content chunks
    thinking_chunks = [c for c in chunks if "thinking" in c["choices"][0]["delta"]]

    assert len(thinking_chunks) >= 1
    assert "Thinking" in thinking_chunks[0]["choices"][0]["delta"]["thinking"]


def test_stream_completion_with_tool_calls_harmony():
    """Test streaming with Harmony tool call format (commentary channel)."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()

    # Mock Harmony format with commentary channel for tool calls
    mock_chunks = [
        {"choices": [{"delta": {"content": "<|channel|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "analysis"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|message|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Use tool"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|end|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "<|channel|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "commentary"}, "finish_reason": None}]},
        {
            "choices": [
                {"delta": {"content": " to=functions.bash"}, "finish_reason": None}
            ]
        },
        {
            "choices": [
                {"delta": {"content": " <|constrain|>json"}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": "<|message|>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '{"cmd":'}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '"ls"}'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    wrapper.model.create_chat_completion.return_value = iter(mock_chunks)

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "List files"}],
        stream=True,
        include_thinking=True,
    )

    chunks = list(result)

    # Find content chunks - should include tool prefix
    content_chunks = [
        c
        for c in chunks
        if "content" in c["choices"][0]["delta"] and c["choices"][0]["delta"]["content"]
    ]

    # Concatenate all content
    full_content = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)

    # Should contain tool prefix and JSON
    assert "to=functions.bash" in full_content
    assert "json" in full_content
    assert '{"cmd":' in full_content or '"ls"}' in full_content


def test_parse_tool_calls_harmony_format():
    """Test parsing Harmony format tool calls."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>commentary to=functions.bash "
        '<|constrain|>json<|message|>{"command":"echo hello"}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "bash"
    assert "echo hello" in result[0]["function"]["arguments"]


def test_parse_tool_calls_harmony_format_with_dot():
    """Test parsing Harmony tool calls with dotted names (container.exec)."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>commentary to=container.exec "
        '<|constrain|>json<|message|>{"cmd":"ls -la"}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "container.exec"


def test_parse_harmony_response_with_final():
    """Test parsing Harmony response with final channel."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>analysis<|message|>Thinking here<|end|>"
        "<|channel|>final<|message|>Final answer<|end|>"
    )

    result = wrapper._parse_harmony_response(content)

    assert result["thinking"] == "Thinking here"
    assert result["content"] == "Final answer"


def test_parse_harmony_response_with_commentary():
    """Test parsing Harmony response with commentary channel (tool calls)."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>analysis<|message|>Need to call tool<|end|>"
        "<|start|>assistant<|channel|>commentary to=bash "
        '<|constrain|>json<|message|>{"cmd":"ls"}<|end|>'
    )

    result = wrapper._parse_harmony_response(content)

    assert result["thinking"] == "Need to call tool"
    # Content should have the tool call JSON after cleaning
    assert "cmd" in result["content"] or "bash" in result["content"]


def test_stream_no_duplicate_content():
    """Test that streamed content isn't duplicated at finish."""
    wrapper = ChatModelWrapper()
    wrapper.model = MagicMock()

    # Simple non-Harmony streaming
    mock_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    wrapper.model.create_chat_completion.return_value = iter(mock_chunks)

    result = wrapper.create_completion(
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = list(result)
    content_chunks = [
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"][0]["delta"].get("content")
    ]

    # Should have exactly "Hello" and " world", no duplicates
    assert content_chunks == ["Hello", " world"]


# =============================================================================
# JSON Repair Tests
# =============================================================================


def test_repair_json_bracket_quote_confusion():
    """Test repairing "] to " (bracket/quote confusion)."""
    wrapper = ChatModelWrapper()
    malformed = '{"command":"git status -s"],"timeout": 120000}'
    expected = '{"command":"git status -s","timeout": 120000}'

    result = wrapper._repair_json(malformed)

    assert result == expected


def test_repair_json_array_with_trailing_comma():
    """Test removing trailing comma inside arrays."""
    wrapper = ChatModelWrapper()
    malformed = '{"flags": ["verbose", "debug",]}'
    expected = '{"flags": ["verbose", "debug"]}'

    result = wrapper._repair_json(malformed)

    assert result == expected


def test_repair_json_trailing_comma():
    """Test removing trailing commas before } or ]."""
    wrapper = ChatModelWrapper()
    malformed = '{"a": 1, "b": 2,}'
    expected = '{"a": 1, "b": 2}'

    result = wrapper._repair_json(malformed)

    assert result == expected


def test_repair_json_multiple_fixes():
    """Test applying multiple repairs in one pass."""
    wrapper = ChatModelWrapper()
    malformed = '{"command":"ls -la"],"timeout": 5000,}'
    expected = '{"command":"ls -la","timeout": 5000}'

    result = wrapper._repair_json(malformed)

    assert result == expected


def test_repair_json_no_changes_needed():
    """Test that valid JSON is returned unchanged."""
    wrapper = ChatModelWrapper()
    valid = '{"command":"git status","timeout": 120000}'

    result = wrapper._repair_json(valid)

    assert result == valid


# =============================================================================
# JSON Repair Integration Tests
# =============================================================================


def test_parse_tool_calls_with_malformed_json_repaired():
    """Test parsing Harmony tool calls with malformed JSON that gets auto-repaired."""
    wrapper = ChatModelWrapper()
    # Malformed: "] instead of "
    content = (
        "<|channel|>commentary to=functions.bash "
        '<|constrain|>json<|message|>{"command":"git status -s"],"timeout": 120000}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "bash"
    # Verify the repaired JSON was parsed correctly
    args = result[0]["function"]["arguments"]
    assert "git status -s" in args
    assert "120000" in args


def test_parse_tool_calls_with_trailing_comma_repaired():
    """Test parsing tool calls with trailing commas that get auto-repaired."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>commentary to=functions.bash "
        '<|constrain|>json<|message|>{"command":"echo hello","timeout": 5000,}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "bash"
    assert "echo hello" in result[0]["function"]["arguments"]


def test_parse_tool_calls_unrepairable_json():
    """Test that completely invalid JSON returns None even after repair attempts."""
    wrapper = ChatModelWrapper()
    content = (
        "<|channel|>commentary to=functions.bash "
        "<|constrain|>json<|message|>{this is not json at all!!!"
    )

    result = wrapper._parse_tool_calls(content)

    assert result is None


# =============================================================================
# Standalone JSON Detection Tests (Format 3)
# =============================================================================


def test_parse_standalone_json_with_command():
    """Test parsing standalone JSON with 'command' field (bash tool)."""
    wrapper = ChatModelWrapper()
    # This is the exact pattern from the OpenAI magic log
    content = '{"command":"ls -R","timeout":120}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "bash"
    assert "ls -R" in result[0]["function"]["arguments"]
    assert "120" in result[0]["function"]["arguments"]


def test_parse_standalone_json_with_text_prefix():
    """Test parsing standalone JSON when there's text before it."""
    wrapper = ChatModelWrapper()
    content = 'Let me run this command: {"command":"git status","timeout":5000}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "bash"
    assert "git status" in result[0]["function"]["arguments"]


def test_parse_standalone_json_with_function_field():
    """Test parsing standalone JSON with 'function' field."""
    wrapper = ChatModelWrapper()
    content = '{"function":"read_file","arguments":{"path":"/tmp/test.txt"}}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "read_file"


def test_parse_standalone_json_with_tool_field():
    """Test parsing standalone JSON with 'tool' field."""
    wrapper = ChatModelWrapper()
    content = '{"tool":"grep","pattern":"TODO","file":"README.md"}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "grep"


def test_parse_standalone_json_no_tool_indicators():
    """Test that standalone JSON without tool indicators returns None."""
    wrapper = ChatModelWrapper()
    # Just a regular JSON object with no tool-related fields
    content = '{"status":"success","count":42,"message":"hello"}'

    result = wrapper._parse_tool_calls(content)

    assert result is None


def test_parse_standalone_json_invalid():
    """Test that invalid standalone JSON returns None."""
    wrapper = ChatModelWrapper()
    content = '{"command":"ls" this is broken}'

    result = wrapper._parse_tool_calls(content)

    assert result is None


def test_parse_prefers_harmony_over_standalone():
    """Test that Harmony format takes precedence over standalone JSON."""
    wrapper = ChatModelWrapper()
    # This content has BOTH Harmony format AND standalone JSON
    # The Harmony format should be detected first
    content = (
        "<|channel|>commentary to=functions.bash "
        '<|constrain|>json<|message|>{"command":"git status"}<|end|> '
        '{"command":"ls -R","timeout":120}'
    )

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    # Should detect the Harmony format tool call (git status), not the standalone one
    assert "git status" in result[0]["function"]["arguments"]


def test_parse_standalone_json_read_tool_with_file_path():
    """Test parsing standalone JSON for read tool with 'file_path' field."""
    wrapper = ChatModelWrapper()
    # This is the exact pattern from the latest OpenAI magic log
    content = '{"file_path":"src/chat.py","offset":0,"limit":400}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "read"
    assert "src/chat.py" in result[0]["function"]["arguments"]
    assert '"offset":0' in result[0]["function"]["arguments"]


def test_parse_standalone_json_read_tool_with_path():
    """Test parsing standalone JSON for read tool with 'path' field."""
    wrapper = ChatModelWrapper()
    content = '{"path":"README.md","limit":100}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "read"
    assert "README.md" in result[0]["function"]["arguments"]


def test_parse_standalone_json_grep_tool():
    """Test parsing standalone JSON for grep tool with 'pattern' field."""
    wrapper = ChatModelWrapper()
    content = '{"pattern":"TODO","file":"src/main.py","output_mode":"content"}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "grep"
    assert "TODO" in result[0]["function"]["arguments"]


def test_parse_standalone_json_grep_with_glob():
    """Test parsing standalone JSON for grep tool with glob parameter."""
    wrapper = ChatModelWrapper()
    content = '{"pattern":"import","glob":"*.py"}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "grep"


def test_parse_standalone_json_web_fetch():
    """Test parsing standalone JSON for web_fetch tool with 'url' field."""
    wrapper = ChatModelWrapper()
    content = '{"url":"https://example.com","prompt":"Get the title"}'

    result = wrapper._parse_tool_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "web_fetch"
    assert "example.com" in result[0]["function"]["arguments"]
