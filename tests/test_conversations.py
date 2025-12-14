"""Tests for Conversations API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.api.openai.conversations import (
    Conversation,
    ConversationItem,
    ConversationManager,
)
from src.config import get_conversations_dir
from src.server import app, get_api_key, settings
from src.util.rate_limiter import get_in_memory_rate_limiter


class TestGetConversationsDir:
    """Test conversation directory path generation."""

    def test_get_conversations_dir_returns_home_path(self):
        """Test that get_conversations_dir returns path under ~/.egemma."""
        result = get_conversations_dir("gpt-oss-20b")
        expected = Path.home() / ".egemma" / "gpt-oss-20b" / "conversations"
        assert result == expected

    def test_get_conversations_dir_different_models(self):
        """Test path varies by model name."""
        path1 = get_conversations_dir("model-a")
        path2 = get_conversations_dir("model-b")
        assert path1 != path2
        assert "model-a" in str(path1)
        assert "model-b" in str(path2)


@pytest.fixture(autouse=True)
def clear_rate_limiter_state():
    """Clear rate limiter state between tests."""
    from src.util.rate_limiter import _client_last_request_time

    _client_last_request_time.clear()


@pytest.fixture
def mock_api_key_dependency():
    """Override get_api_key dependency for testing."""
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"
    yield
    del app.dependency_overrides[get_api_key]


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Provides a TestClient with mocked models and temp conversation_manager."""
    # Disable chat model to prevent loading real GGUF model
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", False)
    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", True)

    async def allow_all_requests(request: Request):
        pass

    app.dependency_overrides[get_in_memory_rate_limiter] = (
        lambda *args, **kwargs: allow_all_requests
    )

    # Use temp directory for conversations to avoid polluting ~/.egemma
    test_conv_dir = tmp_path / "conversations"
    test_conv_dir.mkdir()
    # Patch where it's used (server.py imports it directly)
    monkeypatch.setattr(
        "src.server.get_conversations_dir", lambda model: str(test_conv_dir)
    )

    # Mock the model wrapper classes
    with (
        patch("src.server.SentenceTransformerWrapper") as MockEmbeddingWrapper,
        patch("src.server.SummarizationModelWrapper") as MockSummarizationWrapper,
    ):
        # Setup embedding mock
        mock_embedding_instance = MockEmbeddingWrapper.return_value
        mock_embedding_instance.load_model.return_value = None
        mock_embedding_instance.encode.return_value = np.full(768, 0.1)
        mock_embedding_instance.model = MagicMock()

        # Setup summarization mock
        mock_summarization_instance = MockSummarizationWrapper.return_value
        mock_summarization_instance.load_local_model.return_value = None
        mock_summarization_instance.load_gemini_client.return_value = None
        mock_summarization_instance.summarize.return_value = "This is a summary."
        mock_summarization_instance.model = MagicMock()

        with TestClient(app) as c:
            yield c

    if get_in_memory_rate_limiter in app.dependency_overrides:
        del app.dependency_overrides[get_in_memory_rate_limiter]


HEADERS = {"Authorization": "Bearer test_api_key"}


class TestConversationModel:
    """Test ConversationItem and Conversation data models."""

    def test_conversation_item_to_dict(self):
        """Test ConversationItem serialization."""
        item = ConversationItem(
            id="item_abc123",
            conversation_id="conv_xyz789",
            role="user",
            content="Hello, world!",
            created_at=1234567890.0,
        )
        d = item.to_dict()
        assert d["id"] == "item_abc123"
        assert d["object"] == "conversation.item"
        assert d["conversation_id"] == "conv_xyz789"
        assert d["role"] == "user"
        assert d["content"] == "Hello, world!"
        assert d["created_at"] == 1234567890

    def test_conversation_item_with_tool_calls(self):
        """Test ConversationItem with tool_calls."""
        item = ConversationItem(
            id="item_abc123",
            conversation_id="conv_xyz789",
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )
        d = item.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "call_1"

    def test_conversation_item_with_tool_call_id(self):
        """Test ConversationItem with tool_call_id (tool response)."""
        item = ConversationItem(
            id="item_abc123",
            conversation_id="conv_xyz789",
            role="assistant",
            content="Tool result here",
            tool_call_id="call_1",
        )
        d = item.to_dict()
        assert "tool_call_id" in d
        assert d["tool_call_id"] == "call_1"

    def test_conversation_to_dict(self):
        """Test Conversation serialization."""
        conv = Conversation(
            id="conv_xyz789",
            created_at=1234567890.0,
            metadata={"key": "value"},
        )
        d = conv.to_dict()
        assert d["id"] == "conv_xyz789"
        assert d["object"] == "conversation"
        assert d["created_at"] == 1234567890
        assert d["metadata"] == {"key": "value"}

    def test_conversation_to_list_dict(self):
        """Test Conversation list view serialization."""
        conv = Conversation(
            id="conv_xyz789",
            items=[
                ConversationItem(
                    id="item_1",
                    conversation_id="conv_xyz789",
                    role="user",
                    content="Hello",
                ),
            ],
        )
        d = conv.to_list_dict()
        assert d["id"] == "conv_xyz789"
        assert d["item_count"] == 1
        assert "items" not in d  # List view doesn't include items


class TestConversationManager:
    """Test ConversationManager operations."""

    def test_create_conversation(self, tmp_path):
        """Test creating a conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        assert conv.id.startswith("conv_")
        assert len(conv.items) == 0
        assert (tmp_path / f"{conv.id}.json").exists()

    def test_create_conversation_with_metadata(self, tmp_path):
        """Test creating a conversation with metadata."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create(metadata={"project": "test"})

        assert conv.metadata == {"project": "test"}

    def test_get_conversation(self, tmp_path):
        """Test getting a conversation by ID."""
        manager = ConversationManager(conversations_dir=tmp_path)
        created = manager.create()

        fetched = manager.get(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_conversation_not_found(self, tmp_path):
        """Test getting non-existent conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        fetched = manager.get("conv_nonexistent")
        assert fetched is None

    def test_delete_conversation(self, tmp_path):
        """Test deleting a conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        assert manager.delete(conv.id) is True
        assert manager.get(conv.id) is None
        assert not (tmp_path / f"{conv.id}.json").exists()

    def test_delete_conversation_not_found(self, tmp_path):
        """Test deleting non-existent conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        assert manager.delete("conv_nonexistent") is False

    def test_list_conversations(self, tmp_path):
        """Test listing conversations."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create multiple conversations
        conv1 = manager.create()
        conv2 = manager.create()
        conv3 = manager.create()

        convs = manager.list()
        assert len(convs) == 3

        # Should be sorted by created_at desc
        ids = [c["id"] for c in convs]
        assert conv3.id in ids
        assert conv2.id in ids
        assert conv1.id in ids

    def test_list_conversations_limit(self, tmp_path):
        """Test listing conversations with limit."""
        manager = ConversationManager(conversations_dir=tmp_path)

        for _ in range(5):
            manager.create()

        convs = manager.list(limit=2)
        assert len(convs) == 2

    def test_add_items(self, tmp_path):
        """Test adding items to a conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        items = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        added = manager.add_items(conv.id, items)

        assert len(added) == 2
        assert added[0]["role"] == "user"
        assert added[1]["role"] == "assistant"
        assert added[0]["id"].startswith("item_")

    def test_add_items_conversation_not_found(self, tmp_path):
        """Test adding items to non-existent conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)

        with pytest.raises(ValueError, match="Conversation not found"):
            manager.add_items("conv_nonexistent", [{"role": "user", "content": "Hi"}])

    def test_get_items(self, tmp_path):
        """Test getting items from a conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )

        items = manager.get_items(conv.id)
        assert len(items) == 2
        assert items[0]["content"] == "Hello"
        assert items[1]["content"] == "Hi!"

    def test_get_items_order_asc(self, tmp_path):
        """Test getting items in ascending order."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        )

        items = manager.get_items(conv.id, order="asc")
        assert items[0]["content"] == "First"
        assert items[1]["content"] == "Second"

    def test_get_items_order_desc(self, tmp_path):
        """Test getting items in descending order."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        )

        items = manager.get_items(conv.id, order="desc")
        assert items[0]["content"] == "Second"
        assert items[1]["content"] == "First"

    def test_get_items_limit(self, tmp_path):
        """Test getting items with limit."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [{"role": "user", "content": f"Message {i}"} for i in range(10)],
        )

        items = manager.get_items(conv.id, limit=3)
        assert len(items) == 3

    def test_get_items_as_messages(self, tmp_path):
        """Test getting items formatted as chat messages."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )

        messages = manager.get_items_as_messages(conv.id)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_get_items_as_messages_with_tool_calls(self, tmp_path):
        """Test getting items with tool_calls formatted as chat messages."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conv = manager.create()

        manager.add_items(
            conv.id,
            [
                {"role": "user", "content": "Run a command"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "function": {"name": "bash"}}],
                },
                {"role": "assistant", "content": "Done", "tool_call_id": "call_1"},
            ],
        )

        messages = manager.get_items_as_messages(conv.id)
        assert len(messages) == 3
        assert messages[1]["tool_calls"] == [
            {"id": "call_1", "function": {"name": "bash"}}
        ]
        assert messages[2]["tool_call_id"] == "call_1"

    def test_get_items_nonexistent_conversation(self, tmp_path):
        """Test getting items from non-existent conversation returns empty list."""
        manager = ConversationManager(conversations_dir=tmp_path)
        items = manager.get_items("conv_nonexistent")
        assert items == []

    def test_load_corrupted_json(self, tmp_path):
        """Test loading conversation with corrupted JSON."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Write a corrupted JSON file
        corrupted_file = tmp_path / "conv_corrupted123.json"
        corrupted_file.write_text("{invalid json")

        # Should return None, not raise an exception
        result = manager.get("conv_corrupted123")
        assert result is None

    def test_load_missing_key_json(self, tmp_path):
        """Test loading conversation with missing required key."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Write JSON without required 'id' key
        bad_file = tmp_path / "conv_badkey12345.json"
        bad_file.write_text('{"metadata": {}}')

        # Should return None due to KeyError
        result = manager.get("conv_badkey12345")
        assert result is None

    def test_list_with_corrupted_file(self, tmp_path):
        """Test list() skips corrupted files gracefully."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create a valid conversation
        valid_conv = manager.create()

        # Create a corrupted JSON file
        corrupted = tmp_path / "conv_corrupted12.json"
        corrupted.write_text("{invalid json")

        # list() should return only the valid conversation
        convs = manager.list()
        assert len(convs) == 1
        assert convs[0]["id"] == valid_conv.id


class TestConversationsAPI:
    """Test Conversations API endpoints."""

    def test_create_conversation(self, client, mock_api_key_dependency):
        """Test creating a conversation via API."""
        response = client.post("/v1/conversations", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "conversation"
        assert data["id"].startswith("conv_")

    def test_create_conversation_with_metadata(self, client, mock_api_key_dependency):
        """Test creating a conversation with metadata via API."""
        response = client.post(
            "/v1/conversations",
            headers=HEADERS,
            json={"metadata": {"project": "test"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"] == {"project": "test"}

    def test_list_conversations(self, client, mock_api_key_dependency):
        """Test listing conversations via API."""
        # Create a conversation first
        client.post("/v1/conversations", headers=HEADERS)

        response = client.get("/v1/conversations", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_get_conversation(self, client, mock_api_key_dependency):
        """Test getting a specific conversation via API."""
        # Create conversation
        create_resp = client.post("/v1/conversations", headers=HEADERS)
        conv_id = create_resp.json()["id"]

        # Get it
        response = client.get(f"/v1/conversations/{conv_id}", headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["id"] == conv_id

    def test_get_conversation_not_found(self, client, mock_api_key_dependency):
        """Test getting non-existent conversation via API."""
        response = client.get("/v1/conversations/conv_nonexistent", headers=HEADERS)
        assert response.status_code == 404

    def test_delete_conversation(self, client, mock_api_key_dependency):
        """Test deleting a conversation via API."""
        # Create conversation
        create_resp = client.post("/v1/conversations", headers=HEADERS)
        conv_id = create_resp.json()["id"]

        # Delete it
        response = client.delete(f"/v1/conversations/{conv_id}", headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # Verify deleted
        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=HEADERS)
        assert get_resp.status_code == 404

    def test_delete_conversation_not_found(self, client, mock_api_key_dependency):
        """Test deleting non-existent conversation via API."""
        response = client.delete("/v1/conversations/conv_nonexistent", headers=HEADERS)
        assert response.status_code == 404

    def test_add_and_get_items(self, client, mock_api_key_dependency):
        """Test adding and retrieving items via API."""
        # Create conversation
        create_resp = client.post("/v1/conversations", headers=HEADERS)
        conv_id = create_resp.json()["id"]

        # Add items
        items = {
            "items": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        add_resp = client.post(
            f"/v1/conversations/{conv_id}/items",
            headers=HEADERS,
            json=items,
        )
        assert add_resp.status_code == 200
        assert len(add_resp.json()["data"]) == 2

        # Get items
        get_resp = client.get(
            f"/v1/conversations/{conv_id}/items",
            headers=HEADERS,
        )
        assert get_resp.status_code == 200
        data = get_resp.json()["data"]
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"

    def test_get_items_not_found(self, client, mock_api_key_dependency):
        """Test getting items from non-existent conversation via API."""
        response = client.get(
            "/v1/conversations/conv_nonexistent/items", headers=HEADERS
        )
        assert response.status_code == 404

    def test_add_items_no_body(self, client, mock_api_key_dependency):
        """Test adding items without body via API."""
        # Create conversation
        create_resp = client.post("/v1/conversations", headers=HEADERS)
        conv_id = create_resp.json()["id"]

        # Add items with empty body
        response = client.post(
            f"/v1/conversations/{conv_id}/items",
            headers=HEADERS,
            json={"items": []},
        )
        assert response.status_code == 400

    def test_items_order(self, client, mock_api_key_dependency):
        """Test item ordering via API."""
        # Create and add items
        create_resp = client.post("/v1/conversations", headers=HEADERS)
        conv_id = create_resp.json()["id"]

        items = {
            "items": [{"role": "user", "content": f"Message {i}"} for i in range(5)]
        }
        client.post(
            f"/v1/conversations/{conv_id}/items",
            headers=HEADERS,
            json=items,
        )

        # Get ascending
        asc_resp = client.get(
            f"/v1/conversations/{conv_id}/items?order=asc",
            headers=HEADERS,
        )
        asc_data = asc_resp.json()["data"]
        assert asc_data[0]["content"] == "Message 0"

        # Get descending
        desc_resp = client.get(
            f"/v1/conversations/{conv_id}/items?order=desc",
            headers=HEADERS,
        )
        desc_data = desc_resp.json()["data"]
        assert desc_data[0]["content"] == "Message 4"

    def test_unauthorized_request(self, client):
        """Test request without authorization."""
        response = client.post("/v1/conversations")
        assert response.status_code in [401, 403]
