"""Tests for ChatModelWrapper - GPT-OSS-20B chat completions."""

from unittest.mock import MagicMock, patch

import pytest

from src.chat import ChatModelWrapper


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

    with patch("src.chat.Llama") as mock_llama:
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
    """Test message formatting without tools."""
    wrapper = ChatModelWrapper()
    messages = [{"role": "user", "content": "Hello"}]

    result = wrapper._format_messages_with_tools(messages, None)

    assert result == messages


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
    assert "tool_calls" in result


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

    with patch("src.chat.settings") as mock_settings:
        mock_settings.CHAT_MAX_TOKENS = 2048
        mock_settings.CHAT_TEMPERATURE = 0.5

        wrapper.create_completion(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
        )

        call_args = wrapper.model.create_chat_completion.call_args
        assert call_args.kwargs["max_tokens"] == 2048
        assert call_args.kwargs["temperature"] == 0.5
