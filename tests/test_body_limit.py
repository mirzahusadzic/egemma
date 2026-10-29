"""Request bodies are limited before they are parsed or authenticated."""

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.server import app


@pytest.fixture
def client():
    # No lifespan: the limit must act before any route, model or key is involved.
    return TestClient(app)


def test_an_oversized_upload_is_refused_before_authentication(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 1000)
    body = b"x" * (1000 + 1024 * 1024 + 1)
    response = client.post("/summarize", files={"file": ("big.py", body)})
    assert response.status_code == 413  # not 403: the key is never asked for
    assert "byte limit" in response.json()["detail"]


def test_an_oversized_json_body_is_refused_before_authentication(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_REQUEST_BYTES", 1000)
    response = client.post(
        "/embed/texts",
        content=b'{"texts": ["' + b"x" * 2000 + b'"]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_a_streamed_body_without_a_length_is_cut_off(client, monkeypatch):
    """A chunked body has no Content-Length; the count of bytes received stops it."""
    monkeypatch.setattr(settings, "MAX_REQUEST_BYTES", 1000)

    def chunks():
        for _ in range(50):
            yield b"x" * 100

    response = client.post(
        "/corpus/gpt",
        content=chunks(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert "byte limit" in response.json()["detail"]


def test_a_body_within_the_limit_reaches_the_route(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_REQUEST_BYTES", 1000)
    response = client.post("/embed/texts", json={"texts": ["short"]})
    assert response.status_code == 403  # the route's own authentication answered
