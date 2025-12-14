from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Set a dummy API key for testing
from fastapi import Request
from fastapi.testclient import TestClient

from src.server import EmbeddingDimensions, app, get_api_key, settings
from src.util.rate_limiter import get_in_memory_rate_limiter


@pytest.fixture(autouse=True)
def clear_rate_limiter_state():
    from src.util.rate_limiter import _client_last_request_time

    _client_last_request_time.clear()


@pytest.fixture
def mock_api_key_dependency():
    """
    Overrides the get_api_key dependency to always return a valid key.
    """
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"
    yield
    del app.dependency_overrides[get_api_key]


@pytest.fixture
def client(monkeypatch):
    """Provides a TestClient with mocked models and disabled rate limiting."""
    # Enable local summarization for tests
    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", True)
    # Disable chat model to prevent loading real GGUF model
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", False)

    async def allow_all_requests(request: Request):
        pass

    # Store original dependency to restore it later
    original_get_in_memory_rate_limiter = app.dependency_overrides.get(
        get_in_memory_rate_limiter
    )

    # Override the factory function itself
    app.dependency_overrides[get_in_memory_rate_limiter] = (
        lambda *args, **kwargs: allow_all_requests
    )

    # Apply patches for model loading *before* TestClient is initialized
    with (
        patch("src.server.embedding_model_wrapper.load_model") as mock_embedding_load,
        patch("src.server.embedding_model_wrapper.encode") as mock_embedding_encode,
        patch("src.server.SummarizationModelWrapper") as MockSummarizationWrapper,
    ):
        mock_embedding_load.return_value = None
        mock_embedding_encode.return_value = np.full(768, 0.1)

        mock_summarization_instance = MockSummarizationWrapper.return_value
        mock_summarization_instance.load_local_model.return_value = None
        mock_summarization_instance.load_gemini_client.return_value = None
        mock_summarization_instance.summarize.return_value = "This is a summary."

        # Now create TestClient, so lifespan runs with mocks active
        with TestClient(app) as c:
            yield c

    # Restore original dependency after the test
    if original_get_in_memory_rate_limiter:
        app.dependency_overrides[get_in_memory_rate_limiter] = (
            original_get_in_memory_rate_limiter
        )
    else:
        del app.dependency_overrides[get_in_memory_rate_limiter]


@pytest.fixture
def caching_test_client(monkeypatch):
    """
    Provides a TestClient where the real summarization wrapper is used,
    but the expensive model calls are mocked.
    """
    # Enable local summarization for tests
    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", True)
    # Disable chat model to prevent loading real GGUF model
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", False)

    async def allow_all_requests(request: Request):
        pass

    # Store original dependency to restore it later
    original_get_in_memory_rate_limiter = app.dependency_overrides.get(
        get_in_memory_rate_limiter
    )

    # Override the factory function itself
    app.dependency_overrides[get_in_memory_rate_limiter] = (
        lambda *args, **kwargs: allow_all_requests
    )

    with (
        patch("src.server.embedding_model_wrapper.load_model"),
        patch("llama_cpp.Llama.from_pretrained") as mock_from_pretrained,
    ):
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "This is a cached summary."}}]
        }
        mock_from_pretrained.return_value = mock_llama_instance

        with TestClient(app) as c:
            yield c, mock_llama_instance.create_chat_completion

    # Restore original dependency after the test
    if original_get_in_memory_rate_limiter:
        app.dependency_overrides[get_in_memory_rate_limiter] = (
            original_get_in_memory_rate_limiter
        )
    else:
        del app.dependency_overrides[get_in_memory_rate_limiter]


def test_read_root_redirect(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger-ui-bundle.js" in response.text  # Checks for docs content


def test_embed_no_api_key(client, tmp_path):
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post("/embed", files={"file": ("test.txt", f)})
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}


def test_embed_invalid_api_key(client, tmp_path):
    headers = {"Authorization": "Bearer wrong_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed", files={"file": ("test.txt", f)}, headers=headers
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}


def test_embed_valid_api_key_default_dimensions(
    client, mock_api_key_dependency, tmp_path
):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed", files={"file": ("test.txt", f)}, headers=headers
        )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert len(response.json()["embedding_128d"]) == 128


def test_embed_valid_api_key_specific_dimensions(
    client, mock_api_key_dependency, tmp_path
):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed",
            files={"file": ("test.txt", f)},
            params={
                "dimensions": [
                    EmbeddingDimensions.DIM_256.value,
                    EmbeddingDimensions.DIM_512.value,
                ]
            },
            headers=headers,
        )
    assert response.status_code == 200
    assert "embedding_256d" in response.json()
    assert "embedding_512d" in response.json()
    assert len(response.json()["embedding_256d"]) == 256
    assert len(response.json()["embedding_512d"]) == 512


def test_embed_valid_api_key_all_dimensions(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed",
            files={"file": ("test.txt", f)},
            params={
                "dimensions": [
                    EmbeddingDimensions.DIM_128.value,
                    EmbeddingDimensions.DIM_256.value,
                    EmbeddingDimensions.DIM_512.value,
                    EmbeddingDimensions.DIM_768.value,
                ]
            },
            headers=headers,
        )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert "embedding_256d" in response.json()
    assert "embedding_512d" in response.json()
    assert "embedding_768d" in response.json()
    assert len(response.json()["embedding_128d"]) == 128
    assert len(response.json()["embedding_256d"]) == 256
    assert len(response.json()["embedding_512d"]) == 512
    assert len(response.json()["embedding_768d"]) == 768


def test_embed_valid_api_key_no_dimensions_param(
    client, mock_api_key_dependency, tmp_path
):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed",
            files={"file": ("test.txt", f)},
            headers=headers,
        )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()  # Default behavior
    assert len(response.json()["embedding_128d"]) == 128


def test_encoder_cache_hit(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    text_to_embed = "This is a test sentence for caching."
    (tmp_path / "test.txt").write_text(text_to_embed)

    with patch(
        "src.server.embedding_model_wrapper.encode"
    ) as mock_model_wrapper_encode:
        mock_model_wrapper_encode.return_value = np.full(
            768, 0.1
        )  # Mock the return value

        # Clear the cache before the test to ensure a clean state
        from src.server import PromptNames, cached_encode

        cached_encode.cache_clear()

        # First call
        with open(tmp_path / "test.txt", "rb") as f:
            response1 = client.post(
                "/embed",
                files={"file": ("test.txt", f)},
                headers=headers,
            )
        assert response1.status_code == 200

        # Second call with the same text
        with open(tmp_path / "test.txt", "rb") as f:
            response2 = client.post(
                "/embed",
                files={"file": ("test.txt", f)},
                headers=headers,
            )
        assert response2.status_code == 200

        # Assert that model_wrapper.encode was called only once
        mock_model_wrapper_encode.assert_called_once_with(
            text_to_embed, prompt_name=PromptNames.DOCUMENT, title=None
        )


def test_embed_empty_dimensions(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.txt").write_text("test text")
    with open(tmp_path / "test.txt", "rb") as f:
        response = client.post(
            "/embed",
            files={"file": ("test.txt", f)},
            params={"dimensions": []},
            headers=headers,
        )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert isinstance(response.json()["embedding_128d"], list)


def test_embed_model_not_loaded_error(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    text_to_embed = "test text"
    (tmp_path / "test.txt").write_text(text_to_embed)

    from src.server import cached_encode

    cached_encode.cache_clear()

    with patch("src.server.embedding_model_wrapper.encode") as mock_encode:
        mock_encode.side_effect = RuntimeError("Model not loaded")
        with open(tmp_path / "test.txt", "rb") as f:
            response = client.post(
                "/embed",
                files={"file": ("test.txt", f)},
                headers=headers,
            )
        assert response.status_code == 500, response.text
        assert "Model not loaded" in response.json()["detail"]


def test_summarize_no_api_key(client, tmp_path):
    (tmp_path / "test.py").write_text("def hello(): return 'world'")
    with open(tmp_path / "test.py", "rb") as f:
        response = client.post("/summarize", files={"file": ("test.py", f)})
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}


def test_summarize_invalid_api_key(client, tmp_path):
    headers = {"Authorization": "Bearer wrong_key"}
    (tmp_path / "test.py").write_text("def hello(): return 'world'")
    with open(tmp_path / "test.py", "rb") as f:
        response = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}


def test_summarize_valid_api_key(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.py").write_text("def hello(): return 'world'")
    with open(tmp_path / "test.py", "rb") as f:
        response = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response.status_code == 200
    assert response.json() == {"language": "Python", "summary": "This is a summary."}


def test_summarize_model_not_loaded_error(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.py").write_text("def hello(): return 'world'")

    with patch("src.server.summarization_model_wrapper.summarize") as mock_summarize:
        mock_summarize.side_effect = RuntimeError("Model not loaded")
        with open(tmp_path / "test.py", "rb") as f:
            response = client.post(
                "/summarize", files={"file": ("test.py", f)}, headers=headers
            )
        assert response.status_code == 500, response.text
        assert "Model not loaded" in response.json()["detail"]


def test_summarize_local_disabled(
    client, mock_api_key_dependency, tmp_path, monkeypatch
):
    """Test that local summarization is disabled but Gemini can still be requested."""
    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", False)
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.py").write_text("def hello(): return 'world'")

    # Test that local model request fails
    with open(tmp_path / "test.py", "rb") as f:
        response = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response.status_code == 400
    assert "Local summarization is disabled" in response.json()["detail"]
    assert "Gemini model" in response.json()["detail"]


def test_summarizer_cache_hit(caching_test_client, mock_api_key_dependency, tmp_path):
    client, mock_create_completion = caching_test_client
    headers = {"Authorization": "Bearer test_api_key"}
    file_content = "def hello(): return 'world'  # Unique content for cache test"
    (tmp_path / "test.py").write_text(file_content)

    # Clear the cache on the method before the test
    from src.server import summarization_model_wrapper

    summarization_model_wrapper._summarize_local.cache_clear()

    # First call
    with open(tmp_path / "test.py", "rb") as f:
        response1 = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response1.status_code == 200
    assert response1.json()["summary"] == "This is a cached summary."

    # Second call
    with open(tmp_path / "test.py", "rb") as f:
        response2 = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response2.status_code == 200

    # Assert that the expensive model call was only made once
    mock_create_completion.assert_called_once()


def test_parse_ast_python_success(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    python_code = """\"\"\"A test module for parsing.\"\"\"
import os
from typing import List

class MyClass:
    \"\"\"A simple example class.\"\"\"
    def __init__(self):
        pass

    def my_method(self, arg1: int) -> int:
        return arg1 * 2

def my_function(param1, param2) -> int:
    \"\"\"A standalone function.\"\"\"
    return param1 + param2
"""
    # Create a temporary file with the Python code
    (tmp_path / "test_code.py").write_text(python_code)

    # Post the file to the endpoint
    with open(tmp_path / "test_code.py", "rb") as f:
        response = client.post(
            "/parse-ast",
            files={"file": ("test_code.py", f, "text/x-python-script")},
            data={"language": "python"},
            headers=headers,
        )

    assert response.status_code == 200
    json_response = response.json()

    # Assert top-level structure and imports
    assert json_response["language"] == "python"
    assert json_response["docstring"] == "A test module for parsing."
    assert "os" in json_response["imports"]
    assert "typing" in json_response["imports"]

    # Define the rich, detailed structure the API now returns
    expected_class = {
        "name": "MyClass",
        "docstring": "A simple example class.",
        "base_classes": [],
        "decorators": [],
        "methods": [
            {
                "name": "__init__",
                "docstring": "",
                "params": [{"name": "self", "type": "None"}],
                "returns": "None",
                "decorators": [],
                "is_async": False,
                "body_dependencies": {
                    "instantiations": [],
                    "method_calls": [],
                },
            },
            {
                "name": "my_method",
                "docstring": "",
                "params": [
                    {"name": "self", "type": "None"},
                    {"name": "arg1", "type": "int"},
                ],
                "returns": "int",
                "decorators": [],
                "is_async": False,
                "body_dependencies": {
                    "instantiations": [],
                    "method_calls": [],
                },
            },
        ],
    }
    assert expected_class in json_response["classes"]

    # Define the rich, detailed structure for the standalone function
    expected_function = {
        "name": "my_function",
        "docstring": "A standalone function.",
        "params": [
            {"name": "param1", "type": "None"},
            {"name": "param2", "type": "None"},
        ],
        "returns": "int",
        "decorators": [],
        "is_async": False,
        "body_dependencies": {
            "instantiations": [],
            "method_calls": [],
        },
    }
    assert expected_function in json_response["functions"]


def test_parse_ast_unsupported_language(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    js_code = "console.log('hello');"
    (tmp_path / "test.js").write_text(js_code)
    with open(tmp_path / "test.js", "rb") as f:
        response = client.post(
            "/parse-ast",
            files={"file": ("test.js", f, "text/plain")},
            data={"language": "javascript"},
            headers=headers,
        )
        assert response.status_code == 400
        assert "Language 'javascript' not supported" in response.json()["detail"]


def test_parse_ast_syntax_error(client, mock_api_key_dependency, tmp_path):
    headers = {"Authorization": "Bearer test_api_key"}
    invalid_python_code = "def func(:"
    (tmp_path / "invalid.py").write_text(invalid_python_code)
    with open(tmp_path / "invalid.py", "rb") as f:
        response = client.post(
            "/parse-ast",
            files={"file": ("invalid.py", f, "text/plain")},
            data={"language": "python"},
            headers=headers,
        )
        assert response.status_code == 400
        assert "Python syntax error" in response.json()["detail"]


# =============================================================================
# Tool Call Streaming Suppression Tests (Integration Tests)
# =============================================================================


def test_streaming_suppresses_tool_calls(client, mock_api_key_dependency, monkeypatch):
    """
    Test that tool call content is NOT streamed as text deltas.

    This is a full integration test that:
    1. Mocks the chat model to return tool call chunks
    2. Calls the streaming endpoint
    3. Verifies tool call content is suppressed from the stream
    """
    # Enable chat model for this test
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", True)

    # Create a mock that returns synchronous generator for streaming
    def mock_streaming_completion(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"content": "<|channel|>commentary"}}]},
            {"choices": [{"delta": {"content": " to=functions.bash "}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": '<|constrain|>json<|message|>{"command":"ls"}'
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "<|end|>"}}]},
        ]
        for chunk in chunks:
            yield chunk

    def mock_create_completion(*args, **kwargs):
        # Return synchronous generator if stream=True
        if kwargs.get("stream", False):
            return mock_streaming_completion(*args, **kwargs)
        # Return dict if stream=False
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<|channel|>commentary to=functions.bash "
                            '<|constrain|>json<|message|>{"command":"ls"}<|end|>'
                        )
                    }
                }
            ],
            "usage": {},
        }

    # Create a mock chat model wrapper
    mock_chat = MagicMock()
    mock_chat.is_loaded = True
    mock_chat.create_completion = mock_create_completion
    mock_chat._parse_tool_calls = MagicMock(
        return_value=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ]
    )

    # Patch the global chat_model_wrapper in server.py
    import src.server

    monkeypatch.setattr(src.server, "chat_model_wrapper", mock_chat)

    headers = {"Authorization": "Bearer test_api_key"}
    request_body = {
        "input": [{"type": "message", "role": "user", "content": "List files"}],
        "model": "test-model",
    }

    # Make streaming request (TestClient handles streaming automatically)
    response = client.post("/v1/responses", json=request_body, headers=headers)

    assert response.status_code == 200

    # Collect all events from the stream
    events = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            events.append(decoded)

    # Verify NO text delta events contain tool call markers
    text_delta_events = [e for e in events if "output_text.delta" in e]

    for event in text_delta_events:
        # Tool call markers should NOT appear in text deltas
        assert "<|channel|>commentary" not in event
        assert "to=functions.bash" not in event
        assert "<|constrain|>json" not in event
        assert '{"command":"ls"}' not in event


def test_streaming_allows_regular_text(client, mock_api_key_dependency, monkeypatch):
    """
    Test that regular text content IS streamed normally.

    This verifies that the streaming suppression only affects tool calls,
    not regular text responses.
    """
    # Enable chat model for this test
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", True)

    # Create a mock that returns synchronous generator for streaming
    def mock_streaming_completion(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        for chunk in chunks:
            yield chunk

    def mock_create_completion(*args, **kwargs):
        # Return synchronous generator if stream=True
        if kwargs.get("stream", False):
            return mock_streaming_completion(*args, **kwargs)
        # Return dict if stream=False
        return {
            "choices": [{"message": {"content": "Hello world!"}}],
            "usage": {},
        }

    # Create a mock chat model wrapper
    mock_chat = MagicMock()
    mock_chat.is_loaded = True
    mock_chat.create_completion = mock_create_completion
    mock_chat._parse_tool_calls = MagicMock(return_value=None)

    # Patch the global chat_model_wrapper in server.py
    import src.server

    monkeypatch.setattr(src.server, "chat_model_wrapper", mock_chat)

    headers = {"Authorization": "Bearer test_api_key"}
    request_body = {
        "input": [{"type": "message", "role": "user", "content": "Say hello"}],
        "model": "test-model",
        "stream": True,  # Enable streaming
    }

    # Make streaming request (TestClient handles streaming automatically)
    response = client.post("/v1/responses", json=request_body, headers=headers)

    assert response.status_code == 200

    # Collect all events from the stream
    events = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            events.append(decoded)

    # Verify text delta events ARE present
    text_delta_events = [e for e in events if "output_text.delta" in e]
    assert len(text_delta_events) > 0

    # Verify the content is correct
    import json

    deltas = []
    for event in text_delta_events:
        if "data: " in event:
            data_str = event.split("data: ", 1)[1]
            data = json.loads(data_str)
            if "delta" in data:
                deltas.append(data["delta"])

    # Should contain the regular text
    full_text = "".join(deltas)
    assert "Hello" in full_text or "world" in full_text


def test_streaming_mixed_text_and_tool_calls(
    client, mock_api_key_dependency, monkeypatch
):
    """
    Test that when the model outputs both text and tool calls,
    only the text is streamed (tool calls are suppressed).
    """
    # Enable chat model for this test
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", True)

    # Create a mock that returns synchronous generator for streaming
    def mock_streaming_completion(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"content": "Let me check that. "}}]},
            {"choices": [{"delta": {"content": "<|channel|>commentary"}}]},
            {"choices": [{"delta": {"content": " to=functions.bash "}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": '<|constrain|>json<|message|>{"command":"ls"}'
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "<|end|>"}}]},
            {"choices": [{"delta": {"content": " Done!"}}]},
        ]
        for chunk in chunks:
            yield chunk

    def mock_create_completion(*args, **kwargs):
        # Return synchronous generator if stream=True
        if kwargs.get("stream", False):
            return mock_streaming_completion(*args, **kwargs)
        # Return dict if stream=False
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Let me check that. "
                            "<|channel|>commentary to=functions.bash "
                            '<|constrain|>json<|message|>{"command":"ls"}<|end|> Done!'
                        )
                    }
                }
            ],
            "usage": {},
        }

    # Create a mock chat model wrapper
    mock_chat = MagicMock()
    mock_chat.is_loaded = True
    mock_chat.create_completion = mock_create_completion
    mock_chat._parse_tool_calls = MagicMock(
        return_value=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ]
    )

    # Patch the global chat_model_wrapper in server.py
    import src.server

    monkeypatch.setattr(src.server, "chat_model_wrapper", mock_chat)

    headers = {"Authorization": "Bearer test_api_key"}
    request_body = {
        "input": [{"type": "message", "role": "user", "content": "List files"}],
        "model": "test-model",
        "stream": True,  # Enable streaming
    }

    # Make streaming request (TestClient handles streaming automatically)
    response = client.post("/v1/responses", json=request_body, headers=headers)

    assert response.status_code == 200

    # Collect all events from the stream
    events = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            events.append(decoded)

    # Verify text delta events exist
    text_delta_events = [e for e in events if "output_text.delta" in e]

    import json

    deltas = []
    for event in text_delta_events:
        if "data: " in event:
            data_str = event.split("data: ", 1)[1]
            data = json.loads(data_str)
            if "delta" in data:
                deltas.append(data["delta"])

    full_text = "".join(deltas)

    # Should contain regular text
    assert "Let me check that" in full_text or "Done" in full_text

    # Should NOT contain tool call markers
    assert "<|channel|>commentary" not in full_text
    assert "to=functions.bash" not in full_text
    assert '{"command":"ls"}' not in full_text
