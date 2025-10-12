from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Set a dummy API key for testing
from src.server import EmbeddingDimensions, app, get_api_key, settings


@pytest.fixture
def mock_api_key_dependency():
    """
    Overrides the get_api_key dependency to always return a valid key.
    """
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"
    yield
    del app.dependency_overrides[get_api_key]


@pytest.fixture
def client():
    """
    Provides a TestClient instance with mocked model wrappers.
    """
    with (
        patch("src.server.embedding_model_wrapper.load_model") as mock_embedding_load,
        patch("src.server.embedding_model_wrapper.encode") as mock_embedding_encode,
        patch("src.server.SummarizationModelWrapper") as MockSummarizationWrapper,
    ):
        mock_embedding_load.return_value = None
        mock_embedding_encode.return_value = np.full(768, 0.1)

        mock_summarization_instance = MockSummarizationWrapper.return_value
        mock_summarization_instance.load_model.return_value = None
        mock_summarization_instance.summarize.return_value = "This is a summary."

        with TestClient(app) as c:
            yield c


@pytest.fixture
def caching_test_client():
    """
    Provides a TestClient where the real summarization wrapper is used,
    but the expensive model calls are mocked.
    """
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
        assert response.json() == {"detail": "An internal server error occurred."}


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
        assert response.json() == {"detail": "An internal server error occurred."}


def test_summarize_disabled(client, mock_api_key_dependency, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SUMMARY_ENABLED", False)
    headers = {"Authorization": "Bearer test_api_key"}
    (tmp_path / "test.py").write_text("def hello(): return 'world'")
    with open(tmp_path / "test.py", "rb") as f:
        response = client.post(
            "/summarize", files={"file": ("test.py", f)}, headers=headers
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "Summarization feature is disabled."}


def test_summarizer_cache_hit(caching_test_client, mock_api_key_dependency, tmp_path):
    client, mock_create_completion = caching_test_client
    headers = {"Authorization": "Bearer test_api_key"}
    file_content = "def hello(): return 'world'  # Unique content for cache test"
    (tmp_path / "test.py").write_text(file_content)

    # Clear the cache on the standalone function before the test
    from src.summarization import _cached_summarize

    _cached_summarize.cache_clear()

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
