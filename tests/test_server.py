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
# Session Endpoint Tests
# =============================================================================


@pytest.fixture
def session_client(monkeypatch, tmp_path):
    """Provides a TestClient with mocked models and session manager enabled."""
    from src.session import SessionManager

    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", True)

    async def allow_all_requests(request: Request):
        pass

    app.dependency_overrides[get_in_memory_rate_limiter] = (
        lambda *args, **kwargs: allow_all_requests
    )

    with (
        patch("src.server.embedding_model_wrapper.load_model"),
        patch("src.server.embedding_model_wrapper.encode") as mock_encode,
        patch("src.server.SummarizationModelWrapper") as MockSummarizationWrapper,
        patch("src.server.ChatModelWrapper") as MockChatWrapper,
        patch("src.server.SessionManager") as MockSessionManager,
    ):
        mock_encode.return_value = np.full(768, 0.1)

        mock_summarization_instance = MockSummarizationWrapper.return_value
        mock_summarization_instance.load_local_model.return_value = None

        # Mock chat model
        mock_chat_instance = MockChatWrapper.return_value
        mock_chat_instance.is_loaded = True
        mock_chat_instance.load_model.return_value = None
        mock_chat_instance.model = MagicMock()
        mock_chat_instance.create_completion.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Mock session manager with real behavior
        real_session_manager = SessionManager(
            model=None, max_context=4096, sessions_dir=tmp_path / "sessions"
        )
        MockSessionManager.return_value = real_session_manager

        # Patch the global session_manager in server module
        with patch("src.server.session_manager", real_session_manager):
            with patch("src.server.chat_model_wrapper", mock_chat_instance):
                monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", True)
                with TestClient(app) as c:
                    yield c, real_session_manager

    if get_in_memory_rate_limiter in app.dependency_overrides:
        del app.dependency_overrides[get_in_memory_rate_limiter]


def test_create_session(session_client, mock_api_key_dependency):
    """Test creating a new session."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/sessions", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"].startswith("sess_")
    assert "max_context" in data
    assert "created_at" in data


def test_get_current_session(session_client, mock_api_key_dependency):
    """Test getting current session info."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Create a session first
    create_response = client.post("/v1/sessions", headers=headers)
    session_id = create_response.json()["session_id"]

    # Get current session
    response = client.get("/v1/sessions/current", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "message_count" in data
    assert "token_count" in data


def test_get_current_session_no_session(session_client, mock_api_key_dependency):
    """Test getting current session when none exists."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Don't create a session, directly try to get current
    # Need to reset the session manager state
    session_manager._session = None

    response = client.get("/v1/sessions/current", headers=headers)
    assert response.status_code == 404


def test_load_session(session_client, mock_api_key_dependency):
    """Test loading an existing session."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Create a session
    create_response = client.post("/v1/sessions", headers=headers)
    session_id = create_response.json()["session_id"]

    # Create another session (invalidates first)
    client.post("/v1/sessions", headers=headers)

    # Load the first session
    response = client.post(f"/v1/sessions/{session_id}/load", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id


def test_load_session_not_found(session_client, mock_api_key_dependency):
    """Test loading non-existent session."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/sessions/nonexistent_session/load", headers=headers)
    assert response.status_code == 404


def test_list_sessions(session_client, mock_api_key_dependency):
    """Test listing all sessions."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Create multiple sessions
    client.post("/v1/sessions", headers=headers)
    client.post("/v1/sessions", headers=headers)
    client.post("/v1/sessions", headers=headers)

    response = client.get("/v1/sessions", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert len(data["sessions"]) == 3


def test_get_session_messages(session_client, mock_api_key_dependency):
    """Test getting messages from a session."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Create session and add messages
    create_response = client.post("/v1/sessions", headers=headers)
    session_id = create_response.json()["session_id"]

    # Add messages directly to session manager
    session_manager.add_message({"role": "user", "content": "Hello"})
    session_manager.add_message({"role": "assistant", "content": "Hi there!"})

    response = client.get(f"/v1/sessions/{session_id}/messages", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["message_count"] == 2
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Hello"


def test_get_session_messages_not_found(session_client, mock_api_key_dependency):
    """Test getting messages from non-existent session."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.get("/v1/sessions/nonexistent/messages", headers=headers)
    assert response.status_code == 404


def test_count_tokens(session_client, mock_api_key_dependency):
    """Test token counting endpoint."""
    client, session_manager = session_client
    headers = {"Authorization": "Bearer test_api_key"}

    # Create a session first (needed for session manager to be initialized)
    client.post("/v1/sessions", headers=headers)

    response = client.post(
        "/v1/sessions/count-tokens",
        params={"text": "Hello world, this is a test."},
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "text_length" in data
    assert "token_count" in data
    assert data["text_length"] == 28
