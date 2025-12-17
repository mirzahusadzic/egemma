"""
Conversations API - OpenAI-compatible conversation management.

Provides:
- Conversation CRUD operations (create, get, list, delete)
- Item (message) storage and retrieval
- Disk persistence with JSON files

This API enables 1:1 compatibility with official OpenAI's Conversations API,
allowing cognition-cli to use identical code for both eGemma and api.openai.com.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ConversationItem:
    """A single item (message) in a conversation."""

    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: list[dict] | str  # Support both array (SDK format) and string (legacy)
    created_at: float = field(default_factory=time.time)
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        """Convert item to OpenAI-compatible dictionary."""
        # Normalize content to array format for SDK compatibility
        if isinstance(self.content, str):
            # Convert string to array with single text block
            normalized_content = [{"type": "input_text", "text": self.content}]
        elif isinstance(self.content, list):
            # Already in array format
            normalized_content = self.content
        else:
            # Handle None or other unexpected values - use empty array
            normalized_content = []

        result = {
            "id": self.id,
            "object": "conversation.item",
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": normalized_content,
            "created_at": int(self.created_at),
            "tool_calls": self.tool_calls if self.tool_calls is not None else [],
        }
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: dict, conversation_id: str) -> "ConversationItem":
        """Create ConversationItem from dictionary."""
        # Accept both array and string content
        content = data.get("content", "")
        # Ensure content is never None - normalize to empty string or keep as list
        if content is None:
            content = ""
        elif isinstance(content, list):
            # Already in array format - keep as-is
            pass
        elif not content:
            # Empty string or falsy value
            content = ""

        return cls(
            id=data["id"],
            conversation_id=conversation_id,
            role=data["role"],
            content=content,
            created_at=data.get("created_at", time.time()),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class Conversation:
    """A conversation containing multiple items."""

    id: str
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    items: list[ConversationItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert conversation to OpenAI-compatible dictionary (without items)."""
        return {
            "id": self.id,
            "object": "conversation",
            "created_at": int(self.created_at),
            "metadata": self.metadata,
        }

    def to_dict_with_items(self) -> dict:
        """Convert conversation to dictionary with items for persistence."""
        return {
            **self.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }

    def to_list_dict(self) -> dict:
        """Convert conversation to list-view dictionary (includes item_count)."""
        return {
            "id": self.id,
            "object": "conversation",
            "created_at": int(self.created_at),
            "metadata": self.metadata,
            "item_count": len(self.items),
            "items": [],  # Empty array for SDK compatibility (use get_items to fetch)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Create Conversation from dictionary."""
        conv_id = data["id"]
        items = [
            ConversationItem.from_dict(item, conv_id) for item in data.get("items", [])
        ]
        return cls(
            id=conv_id,
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
            items=items,
        )


class ConversationManager:
    """
    Manages conversations with disk persistence.

    Storage: conversations/{conversation_id}.json

    Thread-safe implementation for concurrent access.
    """

    def __init__(self, conversations_dir: str | Path = "conversations"):
        """
        Initialize conversation manager.

        Args:
            conversations_dir: Directory to store conversation files
        """
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _conv_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation."""
        return self.conversations_dir / f"{conversation_id}.json"

    def _save(self, conv: Conversation) -> None:
        """Save conversation to disk."""
        path = self._conv_path(conv.id)
        with open(path, "w") as f:
            json.dump(conv.to_dict_with_items(), f, indent=2)
        logger.debug(f"Saved conversation {conv.id} to {path}")

    def _load(self, conversation_id: str) -> Conversation | None:
        """Load conversation from disk."""
        path = self._conv_path(conversation_id)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(self, metadata: dict | None = None) -> Conversation:
        """
        Create a new conversation.

        Args:
            metadata: Optional metadata to attach to conversation

        Returns:
            Created Conversation object
        """
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"

        with self._lock:
            conv = Conversation(id=conv_id, metadata=metadata or {})
            self._save(conv)

        logger.info(f"Created conversation: {conv_id}")
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        """
        Get a conversation by ID.

        Args:
            conversation_id: Conversation ID to retrieve

        Returns:
            Conversation object, or None if not found
        """
        with self._lock:
            return self._load(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            path = self._conv_path(conversation_id)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False

    def list(self, limit: int = 20) -> list[dict]:
        """
        List conversations (sorted by created_at desc).

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dictionaries (list view format)
        """
        convs = []
        with self._lock:
            for path in self.conversations_dir.glob("conv_*.json"):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    conv = Conversation.from_dict(data)
                    convs.append(conv.to_list_dict())
                except Exception as e:
                    logger.warning(f"Failed to read conversation {path}: {e}")
                    continue

        # Sort by created_at descending (newest first)
        return sorted(convs, key=lambda c: c["created_at"], reverse=True)[:limit]

    # =========================================================================
    # Item Operations
    # =========================================================================

    def get_items(
        self,
        conversation_id: str,
        limit: int = 100,
        order: str = "asc",
    ) -> list[dict]:
        """
        Get items from a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum items to return
            order: Sort order ("asc" or "desc")

        Returns:
            List of item dictionaries
        """
        with self._lock:
            conv = self._load(conversation_id)
            if conv is None:
                return []

            items = [item.to_dict() for item in conv.items]

            if order == "desc":
                items = items[::-1]

            return items[:limit]

    def add_items(
        self,
        conversation_id: str,
        items: list[dict],
    ) -> list[dict]:
        """
        Add items to a conversation.

        Args:
            conversation_id: Conversation ID
            items: List of item dicts with role/content

        Returns:
            List of created item dictionaries

        Raises:
            ValueError: If conversation not found
        """
        with self._lock:
            conv = self._load(conversation_id)
            if conv is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            added = []
            for item_data in items:
                item_id = f"item_{uuid.uuid4().hex[:12]}"

                # Handle both OpenAI SDK format and direct format
                # SDK may send {"type": "message", "role": "user", ...}
                # or just {"role": "user", "content": "..."}
                role = item_data.get("role")
                if not role and item_data.get("type") == "message":
                    # Extract role from type if present
                    role = "user"  # Default for message type

                if not role:
                    logger.debug(f"Skipping item without role: {item_data}")
                    continue

                # Handle content in both string and array formats
                content = item_data.get("content", "")

                item = ConversationItem(
                    id=item_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,  # Accept both string and array
                    tool_calls=item_data.get("tool_calls"),
                    tool_call_id=item_data.get("tool_call_id"),
                )
                conv.items.append(item)
                added.append(item.to_dict())

            self._save(conv)
            logger.debug(f"Added {len(added)} items to conversation {conversation_id}")

        return added

    def get_items_as_messages(self, conversation_id: str) -> list[dict]:
        """
        Get conversation items formatted as chat messages.

        This is useful for injecting conversation history into
        chat completion requests.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of message dicts with role/content (OpenAI chat format)
        """
        items = self.get_items(conversation_id, limit=10000, order="asc")
        messages = []
        for item in items:
            content = item["content"]
            # For chat completions, convert array content to string if needed
            # (Chat Completions API expects string content, not array)
            if isinstance(content, list) and len(content) > 0:
                # Extract text from first content block
                first_block = content[0]
                if isinstance(first_block, dict):
                    content = first_block.get("text", "")

            msg = {"role": item["role"], "content": content}
            if item.get("tool_calls"):
                msg["tool_calls"] = item["tool_calls"]
            if item.get("tool_call_id"):
                msg["tool_call_id"] = item["tool_call_id"]
            messages.append(msg)
        return messages
