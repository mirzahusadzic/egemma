"""Tests for Harmony format utilities."""

from src.util.harmony_format import (
    format_assistant_message,
    format_conversation,
    format_developer_message,
    format_system_message,
    format_tool_message,
    format_user_message,
)


class TestSystemMessage:
    """Test system message formatting."""

    def test_basic_system_message(self):
        """Test basic system message without functions."""
        result = format_system_message(
            reasoning_effort="high",
            knowledge_cutoff="2024-06",
            current_date="2025-06-28",
            has_functions=False,
        )

        assert result.startswith("<|start|>system<|message|>")
        assert result.endswith("<|end|>")
        assert "You are ChatGPT" in result
        assert "Knowledge cutoff: 2024-06" in result
        assert "Current date: 2025-06-28" in result
        assert "Reasoning: high" in result
        assert "Valid channels: analysis, commentary, final" in result
        assert "functions" not in result  # No function mention when has_functions=False

    def test_system_message_with_functions(self):
        """Test system message with functions enabled."""
        result = format_system_message(has_functions=True)

        assert (
            "Calls to these tools must go to the commentary channel: 'functions'."
            in result
        )


class TestDeveloperMessage:
    """Test developer message formatting."""

    def test_developer_message_with_instructions_only(self):
        """Test developer message with just instructions."""
        result = format_developer_message(instructions="Be helpful and concise")

        assert result.startswith("<|start|>developer<|message|>")
        assert result.endswith("<|end|>")
        assert "# Instructions" in result
        assert "Be helpful and concise" in result

    def test_developer_message_with_tools(self):
        """Test developer message with tool definitions."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = format_developer_message(tools=tools)

        assert "# Tools" in result
        assert "## functions" in result
        assert "namespace functions {" in result
        assert "// Gets the current weather" in result
        assert "type get_weather = (_: {" in result
        assert "location: string," in result
        assert 'format?: "celsius" | "fahrenheit",' in result
        assert "// default: celsius" in result
        assert "}) => any;" in result
        assert "} // namespace functions" in result

    def test_developer_message_with_instructions_and_tools(self):
        """Test developer message with both instructions and tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                },
            }
        ]

        result = format_developer_message(instructions="Be helpful", tools=tools)

        assert "# Instructions" in result
        assert "Be helpful" in result
        assert "# Tools" in result
        assert "type search = () => any;" in result

    def test_tool_without_parameters(self):
        """Test tool definition with no parameters."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_location",
                    "description": "Gets the user's location",
                },
            }
        ]

        result = format_developer_message(tools=tools)

        assert "type get_location = () => any;" in result


class TestUserMessage:
    """Test user message formatting."""

    def test_user_message(self):
        """Test basic user message."""
        result = format_user_message("What is 2 + 2?")

        assert result == "<|start|>user<|message|>What is 2 + 2?<|end|>"


class TestAssistantMessage:
    """Test assistant message formatting."""

    def test_assistant_message_final_channel(self):
        """Test assistant message on final channel."""
        result = format_assistant_message("2 + 2 = 4", channel="final")

        assert result == "<|start|>assistant<|channel|>final<|message|>2 + 2 = 4<|end|>"

    def test_assistant_message_analysis_channel(self):
        """Test assistant message on analysis channel."""
        result = format_assistant_message("Let me think...", channel="analysis")

        assert (
            result
            == "<|start|>assistant<|channel|>analysis<|message|>Let me think...<|end|>"
        )


class TestToolMessage:
    """Test tool message formatting."""

    def test_tool_message(self):
        """Test tool response message."""
        result = format_tool_message(
            "functions.get_weather", '{"temperature": 20, "sunny": true}'
        )

        assert result == (
            "<|start|>functions.get_weather to=assistant<|channel|>commentary"
            '<|message|>{"temperature": 20, "sunny": true}<|end|>'
        )


class TestConversation:
    """Test full conversation formatting."""

    def test_basic_conversation(self):
        """Test formatting a basic conversation."""
        messages = [
            {"role": "user", "content": "Hello!"},
        ]

        result = format_conversation(
            messages=messages,
            instructions="Be friendly",
            reasoning_effort="high",
        )

        # Should contain system message
        assert "<|start|>system<|message|>" in result
        assert "You are ChatGPT" in result
        assert "Reasoning: high" in result

        # Should contain developer message
        assert "<|start|>developer<|message|>" in result
        assert "# Instructions" in result
        assert "Be friendly" in result

        # Should contain user message
        assert "<|start|>user<|message|>Hello!<|end|>" in result

        # Should end with incomplete assistant message
        assert result.endswith("<|start|>assistant")

    def test_conversation_with_tools(self):
        """Test formatting conversation with tools."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            }
        ]

        result = format_conversation(
            messages=messages,
            tools=tools,
        )

        # System message should mention functions channel
        assert (
            "Calls to these tools must go to the commentary channel: 'functions'."
            in result
        )

        # Developer message should have tools section
        assert "# Tools" in result
        assert "namespace functions {" in result
        assert "type get_weather" in result

    def test_conversation_skips_system_role(self):
        """Test that system role messages are skipped in formatted output."""
        messages = [
            {"role": "system", "content": "This should be skipped"},
            {"role": "user", "content": "Hello"},
        ]

        result = format_conversation(
            messages=messages,
            instructions="Be helpful",  # Explicit instructions used
        )

        # System role message should be skipped, not duplicated in output
        # Only one system message (the Harmony system message)
        assert result.count("<|start|>system<|message|>") == 1
        assert "This should be skipped" not in result
        assert "Be helpful" in result  # Explicit instructions used
