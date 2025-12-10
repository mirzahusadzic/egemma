"""Tests for OpenAI Responses API implementation."""

import json

from src.responses import (
    Response,
    ResponseInputItem,
    ResponseOutputFunctionCall,
    ResponseOutputMessage,
    ResponseOutputReasoning,
    ResponseUsage,
    convert_chat_response_to_response,
    convert_tools_to_chat_format,
    create_reasoning_delta_event,
    create_response_created_event,
    create_response_done_event,
    create_stream_event,
    create_text_delta_event,
    generate_call_id,
    generate_message_id,
    generate_reasoning_id,
    generate_response_id,
    parse_input_to_messages,
)


class TestResponseOutputMessage:
    """Test ResponseOutputMessage dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        msg = ResponseOutputMessage(
            id="msg_abc123",
            content=[{"type": "output_text", "text": "Hello"}],
        )
        d = msg.to_dict()
        assert d["id"] == "msg_abc123"
        assert d["type"] == "message"
        assert d["role"] == "assistant"
        assert d["status"] == "completed"
        assert d["content"] == [{"type": "output_text", "text": "Hello"}]

    def test_to_dict_custom_status(self):
        """Test to_dict with custom status."""
        msg = ResponseOutputMessage(
            id="msg_xyz",
            status="in_progress",
        )
        d = msg.to_dict()
        assert d["status"] == "in_progress"


class TestResponseOutputFunctionCall:
    """Test ResponseOutputFunctionCall dataclass."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        fc = ResponseOutputFunctionCall(
            id="msg_fc123",
            call_id="call_abc",
            name="bash",
            arguments='{"command": "ls"}',
        )
        d = fc.to_dict()
        assert d["id"] == "msg_fc123"
        assert d["type"] == "function_call"
        assert d["call_id"] == "call_abc"
        assert d["name"] == "bash"
        assert d["arguments"] == '{"command": "ls"}'
        assert d["status"] == "completed"


class TestResponseOutputReasoning:
    """Test ResponseOutputReasoning dataclass (OpenAI o-series compatible)."""

    def test_to_dict(self):
        """Test to_dict serialization (summary field for SDK)."""
        reasoning = ResponseOutputReasoning(
            id="rs_abc123",
            summary=[{"type": "summary_text", "text": "I need to analyze this..."}],
        )
        d = reasoning.to_dict()
        assert d["id"] == "rs_abc123"
        assert d["type"] == "reasoning"
        assert d["summary"] == [
            {"type": "summary_text", "text": "I need to analyze this..."}
        ]

    def test_to_dict_empty_summary(self):
        """Test to_dict with empty summary."""
        reasoning = ResponseOutputReasoning(id="rs_empty")
        d = reasoning.to_dict()
        assert d["summary"] == []


class TestResponseUsage:
    """Test ResponseUsage dataclass."""

    def test_to_dict(self):
        """Test to_dict serialization (snake_case for OpenAI API)."""
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"] == 150

    def test_to_dict_defaults(self):
        """Test to_dict with default values (snake_case for OpenAI API)."""
        usage = ResponseUsage()
        d = usage.to_dict()
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["total_tokens"] == 0


class TestResponse:
    """Test Response dataclass."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
        )
        d = resp.to_dict()
        assert d["id"] == "resp_123"
        assert d["model"] == "gpt-oss-20b"
        assert d["object"] == "response"
        assert d["status"] == "completed"
        assert d["output"] == []
        assert d["output_text"] == ""
        assert "usage" in d

    def test_to_dict_with_instructions(self):
        """Test to_dict includes instructions when set."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
            instructions="Be helpful",
        )
        d = resp.to_dict()
        assert d["instructions"] == "Be helpful"

    def test_to_dict_with_temperature(self):
        """Test to_dict includes temperature when set."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
            temperature=0.7,
        )
        d = resp.to_dict()
        assert d["temperature"] == 0.7

    def test_to_dict_with_max_output_tokens(self):
        """Test to_dict includes max_output_tokens when set."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
            max_output_tokens=4096,
        )
        d = resp.to_dict()
        assert d["max_output_tokens"] == 4096

    def test_to_dict_with_previous_response_id(self):
        """Test to_dict includes previous_response_id when set."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
            previous_response_id="resp_000",
        )
        d = resp.to_dict()
        assert d["previous_response_id"] == "resp_000"

    def test_to_dict_with_error(self):
        """Test to_dict includes error when set."""
        resp = Response(
            id="resp_123",
            model="gpt-oss-20b",
            error={"code": "rate_limit", "message": "Too many requests"},
        )
        d = resp.to_dict()
        assert d["error"] == {"code": "rate_limit", "message": "Too many requests"}


class TestIdGeneration:
    """Test ID generation functions."""

    def test_generate_response_id_format(self):
        """Test response ID format."""
        rid = generate_response_id()
        assert rid.startswith("resp_")
        assert len(rid) == 29  # resp_ + 24 hex chars

    def test_generate_response_id_unique(self):
        """Test response IDs are unique."""
        ids = [generate_response_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_message_id_format(self):
        """Test message ID format."""
        mid = generate_message_id()
        assert mid.startswith("msg_")
        assert len(mid) == 28  # msg_ + 24 hex chars

    def test_generate_call_id_format(self):
        """Test call ID format."""
        cid = generate_call_id()
        assert cid.startswith("call_")
        assert len(cid) == 29  # call_ + 24 hex chars

    def test_generate_reasoning_id_format(self):
        """Test reasoning ID format (OpenAI o-series compatible)."""
        rid = generate_reasoning_id()
        assert rid.startswith("rs_")
        assert len(rid) == 27  # rs_ + 24 hex chars


class TestParseInputToMessages:
    """Test parse_input_to_messages function."""

    def test_none_input_no_instructions(self):
        """Test with None input and no instructions."""
        messages = parse_input_to_messages(None)
        assert messages == []

    def test_none_input_with_instructions(self):
        """Test with None input but with instructions."""
        messages = parse_input_to_messages(None, instructions="Be helpful")
        assert len(messages) == 1
        assert messages[0] == {"role": "system", "content": "Be helpful"}

    def test_string_input(self):
        """Test with simple string input."""
        messages = parse_input_to_messages("Hello, world!")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello, world!"}

    def test_string_input_with_instructions(self):
        """Test string input with instructions."""
        messages = parse_input_to_messages("Hello", instructions="Be concise")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be concise"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_message_item(self):
        """Test with message type input item."""
        input_data = [{"type": "message", "role": "user", "content": "Hi"}]
        messages = parse_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hi"}

    def test_message_item_default_role(self):
        """Test message item defaults to user role."""
        input_data = [{"type": "message", "content": "Hi"}]
        messages = parse_input_to_messages(input_data)
        assert messages[0]["role"] == "user"

    def test_message_item_with_content_parts(self):
        """Test message with list of content parts."""
        input_data = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                    {"type": "image", "url": "http://..."},  # Should be ignored
                ],
            }
        ]
        messages = parse_input_to_messages(input_data)
        assert messages[0]["content"] == "Part 1\nPart 2"

    def test_function_call_output_item(self):
        """Test with function_call_output type (Harmony format uses 'name')."""
        input_data = [
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "Command executed successfully",
            }
        ]
        messages = parse_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        # Harmony format uses "name" instead of "tool_call_id"
        assert messages[0]["name"] == "unknown"  # No prior function_call to map name
        assert messages[0]["content"] == "Command executed successfully"

    def test_mixed_input_items(self):
        """Test with mixed input types (Harmony format)."""
        input_data = [
            {"type": "message", "role": "user", "content": "Run ls"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": '{"command": "ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "file1.txt\nfile2.txt",
            },
        ]
        messages = parse_input_to_messages(input_data, instructions="You are a CLI")
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        # Harmony format: flat tool_calls without "id", uses "name" directly
        assert messages[2]["tool_calls"][0]["name"] == "bash"
        assert messages[3]["role"] == "tool"
        assert messages[3]["name"] == "bash"

    def test_function_call_item(self):
        """Test function_call creates assistant message with tool_calls."""
        input_data = [
            {"type": "message", "role": "user", "content": "List files"},
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "bash",
                "arguments": '{"command": "ls -la"}',
            },
        ]
        messages = parse_input_to_messages(input_data)
        # Note: function_call without following output or assistant message
        # leaves tool_calls pending - they get flushed at end
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == ""
        assert len(messages[1]["tool_calls"]) == 1
        # Harmony format: flat structure with "name" directly (no nested "function")
        assert messages[1]["tool_calls"][0]["name"] == "bash"
        assert messages[1]["tool_calls"][0]["arguments"] == '{"command": "ls -la"}'

    def test_function_call_output_creates_assistant_if_needed(self):
        """Test function_call_output creates assistant message if no prior one."""
        input_data = [
            {"type": "message", "role": "user", "content": "Run command"},
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "bash",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_xyz",
                "output": "Success",
            },
        ]
        messages = parse_input_to_messages(input_data)
        # user + assistant (with tool_calls) + tool
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        # Harmony format uses "name" instead of "tool_call_id"
        assert messages[2]["name"] == "bash"

    def test_multiple_function_calls(self):
        """Test multiple sequential function calls."""
        input_data = [
            {"type": "message", "role": "user", "content": "Do two things"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result1"},
            {"type": "function_call_output", "call_id": "call_2", "output": "result2"},
        ]
        messages = parse_input_to_messages(input_data)
        # user + assistant (2 tool_calls) + tool + tool
        assert len(messages) == 4
        assert len(messages[1]["tool_calls"]) == 2
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "tool"

    def test_function_call_before_assistant_message(self):
        """Test function_call items attached to following assistant message."""
        input_data = [
            {"type": "message", "role": "user", "content": "Run command"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{}",
            },
            {"type": "message", "role": "assistant", "content": "Ran the command"},
        ]
        messages = parse_input_to_messages(input_data)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Ran the command"
        assert "tool_calls" in messages[1]
        # Harmony format: flat structure with "name" directly (no "id" field)
        assert messages[1]["tool_calls"][0]["name"] == "bash"


class TestConvertToolsToChatFormat:
    """Test convert_tools_to_chat_format function."""

    def test_none_tools(self):
        """Test with None tools."""
        result = convert_tools_to_chat_format(None)
        assert result is None

    def test_empty_tools(self):
        """Test with empty tools list."""
        result = convert_tools_to_chat_format([])
        assert result is None

    def test_function_tool(self):
        """Test converting function tool."""
        tools = [
            {
                "type": "function",
                "name": "bash",
                "description": "Execute bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ]
        result = convert_tools_to_chat_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert result[0]["function"]["description"] == "Execute bash command"

    def test_multiple_tools(self):
        """Test converting multiple tools."""
        tools = [
            {"type": "function", "name": "bash", "description": "Run bash"},
            {"type": "function", "name": "read", "description": "Read file"},
        ]
        result = convert_tools_to_chat_format(tools)
        assert len(result) == 2

    def test_tool_defaults(self):
        """Test tool with missing optional fields uses defaults."""
        tools = [{"type": "function"}]
        result = convert_tools_to_chat_format(tools)
        assert result[0]["function"]["name"] == ""
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"] == {}

    def test_non_function_tool_ignored(self):
        """Test non-function tools are not included."""
        tools = [
            {"type": "code_interpreter"},
            {"type": "function", "name": "bash"},
        ]
        result = convert_tools_to_chat_format(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "bash"


class TestConvertChatResponseToResponse:
    """Test convert_chat_response_to_response function."""

    def test_basic_text_response(self):
        """Test converting basic text response."""
        chat_response = {
            "choices": [{"message": {"content": "Hello, how can I help?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = convert_chat_response_to_response(
            chat_response, response_id="resp_123", model="gpt-oss-20b"
        )
        assert resp.id == "resp_123"
        assert resp.model == "gpt-oss-20b"
        assert resp.output_text == "Hello, how can I help?"
        assert resp.status == "completed"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_response_with_tool_calls(self):
        """Test converting response with tool calls."""
        chat_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "ls"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {},
        }
        resp = convert_chat_response_to_response(
            chat_response, response_id="resp_456", model="gpt-oss-20b"
        )
        assert len(resp.output) == 1
        assert resp.output[0]["type"] == "function_call"
        assert resp.output[0]["name"] == "bash"

    def test_response_with_all_optional_params(self):
        """Test response includes all optional parameters."""
        chat_response = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        resp = convert_chat_response_to_response(
            chat_response,
            response_id="resp_789",
            model="gpt-oss-20b",
            instructions="Be helpful",
            temperature=0.5,
            max_output_tokens=1000,
            previous_response_id="resp_000",
        )
        assert resp.instructions == "Be helpful"
        assert resp.temperature == 0.5
        assert resp.max_output_tokens == 1000
        assert resp.previous_response_id == "resp_000"

    def test_empty_choices(self):
        """Test response with empty choices."""
        chat_response = {"choices": [], "usage": {}}
        resp = convert_chat_response_to_response(
            chat_response, response_id="resp_empty", model="gpt-oss-20b"
        )
        assert resp.output == []
        assert resp.output_text == ""

    def test_tool_call_without_id_generates_one(self):
        """Test tool call without ID gets a generated one."""
        chat_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": "{}"}}
                        ]
                    }
                }
            ],
            "usage": {},
        }
        resp = convert_chat_response_to_response(
            chat_response, response_id="resp_tc", model="gpt-oss-20b"
        )
        assert resp.output[0]["call_id"].startswith("call_")

    def test_response_with_thinking(self):
        """Test converting response with thinking (GPT-OSS -> o-series)."""
        chat_response = {
            "choices": [
                {
                    "message": {
                        "thinking": "Let me analyze this step by step...",
                        "content": "Here is my answer.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        }
        resp = convert_chat_response_to_response(
            chat_response, response_id="resp_think", model="gpt-oss-20b"
        )
        # Reasoning should be first in output
        assert len(resp.output) == 2
        assert resp.output[0]["type"] == "reasoning"
        assert resp.output[0]["id"].startswith("rs_")
        assert (
            resp.output[0]["summary"][0]["text"]
            == "Let me analyze this step by step..."
        )
        # Message should be second
        assert resp.output[1]["type"] == "message"
        assert resp.output_text == "Here is my answer."


class TestStreamEventCreation:
    """Test streaming event creation functions."""

    def test_create_stream_event(self):
        """Test basic stream event creation."""
        event = create_stream_event("test.event", {"key": "value"}, "resp_123")
        assert event.startswith("event: test.event\n")
        assert "data: " in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["type"] == "test.event"
        assert data["response_id"] == "resp_123"
        assert data["key"] == "value"

    def test_create_response_created_event(self):
        """Test response.created event."""
        resp = Response(id="resp_abc", model="gpt-oss-20b")
        event = create_response_created_event(resp)
        assert "event: response.created" in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["response"]["id"] == "resp_abc"

    def test_create_text_delta_event(self):
        """Test response.output_text.delta event."""
        event = create_text_delta_event("resp_abc", "Hello", "item_123")
        assert "event: response.output_text.delta" in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["delta"] == "Hello"
        assert data["item_id"] == "item_123"

    def test_create_reasoning_delta_event(self):
        """Test response.reasoning_summary.delta event."""
        event = create_reasoning_delta_event(
            "resp_xyz", "Analyzing the problem...", "rs_123"
        )
        assert "event: response.reasoning_summary.delta" in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["delta"] == "Analyzing the problem..."
        assert data["item_id"] == "rs_123"

    def test_create_response_done_event(self):
        """Test response.completed event."""
        resp = Response(id="resp_xyz", model="gpt-oss-20b", output_text="Done!")
        event = create_response_done_event(resp)
        assert "event: response.completed" in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["response"]["output_text"] == "Done!"


class TestResponseInputItem:
    """Test ResponseInputItem dataclass."""

    def test_message_type(self):
        """Test message type input item."""
        item = ResponseInputItem(type="message", role="user", content="Hello")
        assert item.type == "message"
        assert item.role == "user"
        assert item.content == "Hello"

    def test_item_reference_type(self):
        """Test item_reference type input item."""
        item = ResponseInputItem(type="item_reference", item_id="item_abc")
        assert item.type == "item_reference"
        assert item.item_id == "item_abc"
