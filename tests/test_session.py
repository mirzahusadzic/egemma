"""Tests for SessionManager - persistent chat session management."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.session import Session, SessionManager


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = Session(session_id="test_123")
        assert session.session_id == "test_123"
        assert session.messages == []
        assert session.token_count == 0
        assert session.max_context == 65536

    def test_session_tokens_remaining(self):
        """Test tokens_remaining property."""
        session = Session(session_id="test", token_count=1000, max_context=4096)
        assert session.tokens_remaining == 3096

    def test_session_tokens_remaining_at_limit(self):
        """Test tokens_remaining when at context limit."""
        session = Session(session_id="test", token_count=4096, max_context=4096)
        assert session.tokens_remaining == 0

    def test_session_tokens_remaining_over_limit(self):
        """Test tokens_remaining when over context limit returns 0."""
        session = Session(session_id="test", token_count=5000, max_context=4096)
        assert session.tokens_remaining == 0

    def test_session_context_usage(self):
        """Test context_usage percentage calculation."""
        session = Session(session_id="test", token_count=2048, max_context=4096)
        assert session.context_usage == 0.5

    def test_session_context_usage_zero_max(self):
        """Test context_usage with zero max_context."""
        session = Session(session_id="test", token_count=100, max_context=0)
        assert session.context_usage == 0

    def test_session_to_dict(self):
        """Test session serialization to dictionary."""
        session = Session(
            session_id="test_123",
            messages=[{"role": "user", "content": "Hello"}],
            token_count=10,
            max_context=4096,
        )
        data = session.to_dict()
        assert data["session_id"] == "test_123"
        assert data["messages"] == [{"role": "user", "content": "Hello"}]
        assert data["token_count"] == 10
        assert data["max_context"] == 4096

    def test_session_to_stats(self):
        """Test session stats (without messages)."""
        session = Session(
            session_id="test_123",
            messages=[{"role": "user", "content": "Hello"}],
            token_count=10,
            max_context=4096,
        )
        stats = session.to_stats()
        assert stats["session_id"] == "test_123"
        assert stats["message_count"] == 1
        assert stats["token_count"] == 10
        assert "messages" not in stats

    def test_session_from_dict(self):
        """Test session deserialization from dictionary."""
        data = {
            "session_id": "test_123",
            "messages": [{"role": "user", "content": "Hello"}],
            "token_count": 10,
            "max_context": 4096,
            "created_at": 1234567890.0,
            "last_accessed": 1234567890.0,
        }
        session = Session.from_dict(data)
        assert session.session_id == "test_123"
        assert len(session.messages) == 1
        assert session.token_count == 10


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary directory for session files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        return SessionManager(
            model=None, max_context=4096, sessions_dir=temp_sessions_dir
        )

    @pytest.fixture
    def manager_with_model(self, temp_sessions_dir):
        """Create SessionManager with mocked model."""
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        return SessionManager(
            model=mock_model, max_context=4096, sessions_dir=temp_sessions_dir
        )

    def test_manager_init(self, temp_sessions_dir):
        """Test SessionManager initialization."""
        manager = SessionManager(sessions_dir=temp_sessions_dir)
        assert manager.model is None
        assert manager.max_context == 65536
        assert manager._session is None

    def test_manager_init_creates_dir(self, temp_sessions_dir):
        """Test SessionManager creates sessions directory."""
        new_dir = temp_sessions_dir / "new_sessions"
        SessionManager(sessions_dir=new_dir)
        assert new_dir.exists()

    def test_set_model(self, manager):
        """Test setting model after initialization."""
        mock_model = MagicMock()
        manager.set_model(mock_model)
        assert manager.model is mock_model

    def test_create_session(self, manager):
        """Test creating a new session."""
        session = manager.create_session()
        assert session is not None
        assert session.session_id.startswith("sess_")
        assert session.max_context == 4096
        assert manager._session is session

    def test_create_session_saves_to_disk(self, manager, temp_sessions_dir):
        """Test that create_session saves session file."""
        session = manager.create_session()
        session_file = temp_sessions_dir / f"{session.session_id}.json"
        assert session_file.exists()

    def test_create_session_invalidates_previous(self, manager):
        """Test creating new session invalidates previous."""
        session1 = manager.create_session()
        session1_id = session1.session_id
        session2 = manager.create_session()
        assert manager._session.session_id == session2.session_id
        assert session2.session_id != session1_id

    def test_load_session(self, manager, temp_sessions_dir):
        """Test loading a session from disk."""
        # Create and save a session
        session = manager.create_session()
        session_id = session.session_id
        manager.add_message({"role": "user", "content": "Hello"})

        # Create new manager and load session
        manager2 = SessionManager(sessions_dir=temp_sessions_dir, max_context=4096)
        loaded = manager2.load_session(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Hello"

    def test_load_session_not_found(self, manager):
        """Test loading non-existent session returns None."""
        result = manager.load_session("nonexistent_session")
        assert result is None

    def test_load_session_returns_current_if_same_id(self, manager):
        """Test loading current session returns it directly."""
        session = manager.create_session()
        session_id = session.session_id
        loaded = manager.load_session(session_id)
        assert loaded is session

    def test_get_session_no_active(self, manager):
        """Test get_session returns None when no active session."""
        assert manager.get_session() is None

    def test_get_session_returns_current(self, manager):
        """Test get_session returns current session."""
        session = manager.create_session()
        assert manager.get_session() is session

    def test_get_session_by_id(self, manager, temp_sessions_dir):
        """Test get_session with specific ID loads it."""
        session1 = manager.create_session()
        session1_id = session1.session_id
        manager.add_message({"role": "user", "content": "First"})

        manager.create_session()
        manager.add_message({"role": "user", "content": "Second"})

        # Get session1 by ID (should load it)
        loaded = manager.get_session(session1_id)
        assert loaded.session_id == session1_id

    def test_get_messages_empty(self, manager):
        """Test get_messages returns empty list when no session."""
        assert manager.get_messages() == []

    def test_get_messages(self, manager):
        """Test get_messages returns session messages."""
        manager.create_session()
        manager.add_message({"role": "user", "content": "Hello"})
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_get_messages_returns_copy(self, manager):
        """Test get_messages returns a copy, not reference."""
        manager.create_session()
        manager.add_message({"role": "user", "content": "Hello"})
        messages = manager.get_messages()
        messages.append({"role": "user", "content": "Modified"})
        assert len(manager.get_messages()) == 1

    def test_add_message_no_session(self, manager):
        """Test add_message raises error when no session."""
        with pytest.raises(ValueError, match="No active session"):
            manager.add_message({"role": "user", "content": "Hello"})

    def test_add_message(self, manager):
        """Test adding a message to session."""
        manager.create_session()
        tokens = manager.add_message({"role": "user", "content": "Hello"})
        assert tokens > 0
        assert len(manager._session.messages) == 1

    def test_add_message_updates_token_count(self, manager_with_model):
        """Test add_message updates session token count."""
        manager_with_model.create_session()
        initial_tokens = manager_with_model._session.token_count
        manager_with_model.add_message({"role": "user", "content": "Hello"})
        assert manager_with_model._session.token_count > initial_tokens

    def test_add_messages(self, manager):
        """Test adding multiple messages."""
        manager.create_session()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        total_tokens = manager.add_messages(messages)
        assert total_tokens > 0
        assert len(manager._session.messages) == 2

    def test_add_messages_no_session(self, manager):
        """Test add_messages raises error when no session."""
        with pytest.raises(ValueError, match="No active session"):
            manager.add_messages([{"role": "user", "content": "Hello"}])

    def test_count_tokens_with_model(self, manager_with_model):
        """Test token counting with model tokenizer."""
        count = manager_with_model.count_tokens("Hello world")
        assert count == 5  # Mocked to return 5 tokens

    def test_count_tokens_without_model(self, manager):
        """Test token counting fallback without model."""
        count = manager.count_tokens("Hello world!")  # 12 chars
        assert count == 3  # 12 // 4 = 3

    def test_count_tokens_tokenizer_error(self, manager_with_model):
        """Test token counting fallback on tokenizer error."""
        manager_with_model.model.tokenize.side_effect = Exception("Tokenizer error")
        count = manager_with_model.count_tokens("Hello world!")
        assert count == 3  # Falls back to estimate

    def test_check_context_available_no_session(self, manager):
        """Test check_context_available with no session."""
        has_space, remaining = manager.check_context_available(100)
        assert has_space is False
        assert remaining == 0

    def test_check_context_available_has_space(self, manager):
        """Test check_context_available when space available."""
        manager.create_session()
        has_space, remaining = manager.check_context_available(100)
        assert has_space is True
        assert remaining == 4096

    def test_check_context_available_no_space(self, manager):
        """Test check_context_available when no space."""
        manager.create_session()
        manager._session.token_count = 4000
        has_space, remaining = manager.check_context_available(200)
        assert has_space is False
        assert remaining == 96

    def test_list_sessions_empty(self, manager):
        """Test list_sessions with no saved sessions."""
        sessions = manager.list_sessions()
        assert sessions == []

    def test_list_sessions(self, manager):
        """Test list_sessions returns all saved sessions."""
        manager.create_session()
        manager.create_session()
        manager.create_session()
        sessions = manager.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_sorted_by_last_accessed(self, manager):
        """Test list_sessions returns sessions sorted by last_accessed."""
        manager.create_session()
        time.sleep(0.01)
        manager.create_session()
        time.sleep(0.01)
        latest_session = manager.create_session()

        sessions = manager.list_sessions()
        # Most recently accessed first
        assert sessions[0]["session_id"] == latest_session.session_id

    def test_get_stats_no_session(self, manager):
        """Test get_stats with no active session."""
        stats = manager.get_stats()
        assert stats["has_session"] is False
        assert stats["session"] is None

    def test_get_stats_with_session(self, manager):
        """Test get_stats with active session."""
        session = manager.create_session()
        stats = manager.get_stats()
        assert stats["has_session"] is True
        assert stats["session"]["session_id"] == session.session_id

    def test_session_persistence_across_managers(self, temp_sessions_dir):
        """Test sessions persist across manager instances."""
        # Create session with manager 1
        manager1 = SessionManager(sessions_dir=temp_sessions_dir, max_context=4096)
        session = manager1.create_session()
        session_id = session.session_id
        manager1.add_message({"role": "user", "content": "Hello"})
        manager1.add_message({"role": "assistant", "content": "Hi!"})

        # Load with manager 2
        manager2 = SessionManager(sessions_dir=temp_sessions_dir, max_context=4096)
        loaded = manager2.load_session(session_id)

        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["content"] == "Hi!"

    def test_corrupted_session_file(self, manager, temp_sessions_dir):
        """Test loading corrupted session file returns None."""
        # Create corrupted session file
        bad_file = temp_sessions_dir / "sess_corrupted.json"
        bad_file.write_text("not valid json{{{")

        result = manager.load_session("sess_corrupted")
        assert result is None
