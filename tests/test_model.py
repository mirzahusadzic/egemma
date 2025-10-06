import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Set a dummy API key for testing
os.environ["WORKBENCH_API_KEY"] = "test_api_key"

# Import the app after setting the environment variable
from model import app, EmbeddingDimensions

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """
    Mocks the SentenceTransformer to prevent actual model loading during tests.
    """
    with patch("model.SentenceTransformer") as mock_st:
        # Configure the mock to return a dummy embedding
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [0.1] * 768  # A dummy 768-dim embedding
        mock_st.return_value = mock_instance
        yield mock_st

def test_read_root_redirect():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger-ui-bundle.js" in response.text # Checks for docs content

def test_embed_no_api_key():
    response = client.post("/embed", json={"text": "test text"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}

def test_embed_invalid_api_key():
    headers = {"Authorization": "Bearer wrong_key"}
    response = client.post("/embed", json={"text": "test text"}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}

def test_embed_valid_api_key_default_dimensions():
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post("/embed", json={"text": "test text"}, headers=headers)
    assert response.status_code == 200
    assert "embedding_128d" in response.json()
    assert len(response.json()["embedding_128d"]) == 128

def test_embed_valid_api_key_specific_dimensions():
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        params={"dimensions": [EmbeddingDimensions.DIM_256, EmbeddingDimensions.DIM_512]},
        headers=headers,
    )
    assert response.status_code == 200
    assert "embedding_256d" in response.json()
    assert "embedding_512d" in response.json()
    assert len(response.json()["embedding_256d"]) == 256
    assert len(response.json()["embedding_512d"]) == 512

def test_embed_valid_api_key_all_dimensions():
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        params={"dimensions": [
            EmbeddingDimensions.DIM_128,
            EmbeddingDimensions.DIM_256,
            EmbeddingDimensions.DIM_512,
            EmbeddingDimensions.DIM_768,
        ]},
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

def test_embed_valid_api_key_no_dimensions_param():
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "embedding_128d" in response.json() # Default behavior
    assert len(response.json()["embedding_128d"]) == 128

def test_embed_valid_api_key_empty_dimensions_list():
    headers = {"Authorization": "Bearer test_api_key"}
    response = client.post(
        "/embed",
        json={"text": "test text"},
        params={"dimensions": []},
        headers=headers,
    )
    assert response.status_code == 200
    assert "embedding" in response.json() # Should return full embedding if list is empty
    assert len(response.json()["embedding"]) == 768 # Assuming 768 is the full dimension
