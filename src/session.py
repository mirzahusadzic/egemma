"""
Session Manager for GPT-OSS-20B Chat Completions

Persistent session storage with:
- Message history saved to disk (matches Claude/Gemini behavior)
- Token counting using llama-cpp tokenizer
- Resume old sessions by ID
- Single active session (loading new/old invalidates current)
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a chat session with stored messages."""

    session_id: str
    messages: list[dict] = field(default_factory=list)
    token_count: int = 0
    max_context: int = 65536
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def tokens_remaining(self) -> int:
        """Tokens available before context limit."""
        return max(0, self.max_context - self.token_count)

    @property
    def context_usage(self) -> float:
        """Context usage as percentage (0.0 - 1.0)."""
        return self.token_count / self.max_context if self.max_context > 0 else 0

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "token_count": self.token_count,
            "max_context": self.max_context,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
        }

    def to_stats(self) -> dict:
        """Convert session to stats (without messages)."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "token_count": self.token_count,
            "max_context": self.max_context,
            "tokens_remaining": self.tokens_remaining,
            "context_usage": round(self.context_usage, 4),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create Session from dictionary."""
        return cls(
            session_id=data["session_id"],
            messages=data.get("messages", []),
            token_count=data.get("token_count", 0),
            max_context=data.get("max_context", 65536),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
        )


class SessionManager:
    """
    Persistent session manager with disk storage.

    Matches Claude/Gemini behavior:
    - Messages stored server-side and persisted to disk
    - create_session() → new session (saves + invalidates old)
    - load_session(id) → resume old session (invalidates current)
    - Single active session at a time
    """

    def __init__(
        self,
        model: "Llama | None" = None,
        max_context: int = 65536,
        sessions_dir: str | Path = "sessions",
    ):
        """
        Initialize session manager.

        Args:
            model: Llama model instance for tokenization
            max_context: Maximum context window size
            sessions_dir: Directory to store session files
        """
        self.model = model
        self.max_context = max_context
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self._session: Session | None = None
        self._lock = threading.RLock()

    def set_model(self, model: "Llama") -> None:
        """Set the model for tokenization."""
        self.model = model

    def _session_path(self, session_id: str) -> Path:
        """Get path for session file."""
        return self.sessions_dir / f"{session_id}.json"

    def _save_session(self, session: Session) -> None:
        """Save session to disk."""
        path = self._session_path(session.session_id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        logger.debug(f"Saved session to {path}")

    def _load_session_from_disk(self, session_id: str) -> Session | None:
        """Load session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def create_session(self) -> Session:
        """
        Create a new clean session.

        Saves current session to disk before invalidating.
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        with self._lock:
            # Save and invalidate current session
            if self._session:
                self._save_session(self._session)
                logger.info(
                    f"Saved and invalidated session {self._session.session_id} "
                    f"(messages: {len(self._session.messages)}, "
                    f"tokens: {self._session.token_count})"
                )

            self._session = Session(
                session_id=session_id,
                max_context=self.max_context,
            )
            # Save new session immediately
            self._save_session(self._session)

        logger.info(f"Created session: {session_id}")
        return self._session

    def load_session(self, session_id: str) -> Session | None:
        """
        Load/resume an old session by ID.

        Saves current session before loading the requested one.
        If session_id matches current, just returns current.

        Args:
            session_id: Session ID to load

        Returns:
            Loaded session, or None if not found
        """
        with self._lock:
            # If requesting current session, just return it
            if self._session and self._session.session_id == session_id:
                self._session.last_accessed = time.time()
                return self._session

            # Save current session before switching
            if self._session:
                self._save_session(self._session)
                logger.info(f"Saved session {self._session.session_id} before switch")

            # Load requested session
            loaded = self._load_session_from_disk(session_id)
            if loaded is None:
                logger.warning(f"Session not found: {session_id}")
                return None

            loaded.last_accessed = time.time()
            self._session = loaded
            logger.info(
                f"Loaded session {session_id} "
                f"(messages: {len(loaded.messages)}, tokens: {loaded.token_count})"
            )
            return self._session

    def get_session(self, session_id: str | None = None) -> Session | None:
        """
        Get session - current or by ID.

        If session_id provided and differs from current, attempts to load it.
        """
        with self._lock:
            if self._session is None:
                return None

            # If no ID specified, return current
            if session_id is None:
                self._session.last_accessed = time.time()
                return self._session

            # If ID matches current, return it
            if self._session.session_id == session_id:
                self._session.last_accessed = time.time()
                return self._session

            # Otherwise, try to load the requested session
            return self.load_session(session_id)

    def get_messages(self, session_id: str | None = None) -> list[dict]:
        """Get stored messages from session."""
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.messages.copy()

    def add_message(self, message: dict) -> int:
        """
        Add a message to current session.

        Args:
            message: Message dict with 'role' and 'content'

        Returns:
            Token count of added message
        """
        with self._lock:
            if self._session is None:
                raise ValueError("No active session")

            msg_tokens = self._count_message_tokens(message)
            self._session.messages.append(message)
            self._session.token_count += msg_tokens
            self._session.last_accessed = time.time()

            # Auto-save after each message
            self._save_session(self._session)

            return msg_tokens

    def add_messages(self, messages: list[dict]) -> int:
        """
        Add multiple messages to current session.

        Args:
            messages: List of message dicts

        Returns:
            Total token count of added messages
        """
        total_tokens = 0
        with self._lock:
            if self._session is None:
                raise ValueError("No active session")

            for msg in messages:
                msg_tokens = self._count_message_tokens(msg)
                self._session.messages.append(msg)
                self._session.token_count += msg_tokens
                total_tokens += msg_tokens

            self._session.last_accessed = time.time()
            # Save once after all messages added
            self._save_session(self._session)

        return total_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model tokenizer."""
        if self.model is None:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using estimate")
            return len(text) // 4

    def _count_message_tokens(self, message: dict) -> int:
        """Count tokens in a single message including overhead."""
        # ChatML format overhead per message: ~4 tokens
        overhead = 4
        content = message.get("content", "")
        return self.count_tokens(content) + overhead

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens in a list of messages."""
        return sum(self._count_message_tokens(m) for m in messages)

    def check_context_available(self, required_tokens: int) -> tuple[bool, int]:
        """
        Check if context is available for request.

        Returns:
            Tuple of (has_space, tokens_remaining)
        """
        with self._lock:
            if self._session is None:
                return False, 0

            available = self._session.tokens_remaining
            return required_tokens <= available, available

    def list_sessions(self) -> list[dict]:
        """List all saved sessions (stats only, no messages)."""
        sessions = []
        for path in self.sessions_dir.glob("sess_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                session = Session.from_dict(data)
                sessions.append(session.to_stats())
            except Exception as e:
                logger.warning(f"Failed to read session {path}: {e}")
        return sorted(sessions, key=lambda s: s["last_accessed"], reverse=True)

    def get_stats(self) -> dict:
        """Get session manager statistics."""
        with self._lock:
            if self._session:
                return {
                    "has_session": True,
                    "session": self._session.to_stats(),
                    "sessions_dir": str(self.sessions_dir),
                }
            return {
                "has_session": False,
                "session": None,
                "sessions_dir": str(self.sessions_dir),
            }
