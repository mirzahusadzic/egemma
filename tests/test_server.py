from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Set a dummy API key for testing
from src.server import EmbeddingDimensions, app, get_api_key


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
    Provides a TestClient instance with a mocked SentenceTransformer.
    """
    with (
        patch("src.server.model_wrapper.load_model") as mock_load,
        patch("src.server.model_wrapper.encode") as mock_encode,
    ):
        mock_load.return_value = None
        mock_encode.return_value = np.full(768, 0.1)
        with TestClient(app) as c:
            yield c


def test_read_root_redirect(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger-ui-bundle.js" in response.text  # Checks for docs content


def test_embed_no_api_key(client):
    response = client.post("/embed", json={"text": "test text"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}


def test_embed_invalid_api_key(client):
    headers = {"Authorization": "Bearer wrong_key"}
    response = client.post("/embed", json={"text": "test text"}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}


def test_embed_valid_api_key_default_dimensions(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post("/embed", json={"text": "test text"}, headers=headers)
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert len(response.json()["embedding_128d"]) == 128


def test_embed_valid_api_key_specific_dimensions(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
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


def test_embed_valid_api_key_all_dimensions(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
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


def test_embed_valid_api_key_no_dimensions_param(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()  # Default behavior
    assert len(response.json()["embedding_128d"]) == 128


def test_encoder_cache_hit(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    text_to_embed = "This is a test sentence for caching."

    with patch("src.server.model_wrapper.encode") as mock_model_wrapper_encode:
        mock_model_wrapper_encode.return_value = np.full(
            768, 0.1
        )  # Mock the return value

        # Clear the cache before the test to ensure a clean state
        from src.server import cached_encode

        cached_encode.cache_clear()

        # First call
        response1 = client.post(
            "/embed",
            json={"text": text_to_embed},
            headers=headers,
        )
        assert response1.status_code == 200

        # Second call with the same text
        response2 = client.post(
            "/embed",
            json={"text": text_to_embed},
            headers=headers,
        )
        assert response2.status_code == 200

        # Assert that model_wrapper.encode was called only once
        mock_model_wrapper_encode.assert_called_once_with(text_to_embed)


def test_embed_empty_dimensions(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        params={"dimensions": []},
        headers=headers,
    )
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert isinstance(response.json()["embedding_128d"], list)


def test_embed_model_not_loaded_error(client, mock_api_key_dependency):
    headers = {"Authorization": "Bearer test_api_key"}
    text_to_embed = "test text"

    from src.server import cached_encode

    cached_encode.cache_clear()

    with patch("src.server.model_wrapper.encode") as mock_encode:
        mock_encode.side_effect = RuntimeError("Model not loaded")
        response = client.post(
            "/embed",
            json={"text": text_to_embed},
            headers=headers,
        )
        assert response.status_code == 500, response.text
        assert response.json() == {"detail": "Model not loaded"}
