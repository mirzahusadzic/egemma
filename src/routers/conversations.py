import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..api.openai.conversations import ConversationManager
from ..config import settings
from ..util.rate_limiter import get_in_memory_rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared conversation manager instance (initialized in server.py)
conversation_manager: ConversationManager | None = None

# Create router
router = APIRouter(tags=["Conversations"], prefix="/v1")


@router.post(
    "/conversations",
    summary="Create Conversation",
    description="Create a new conversation.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def create_conversation(
    request: Request,
):
    """Create a new conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    # Parse optional metadata from request body
    metadata = None
    try:
        body = await request.json()
        metadata = body.get("metadata")
    except Exception:
        pass  # No body or invalid JSON is OK

    conv = conversation_manager.create(metadata)
    return conv.to_dict()


@router.get(
    "/conversations",
    summary="List Conversations",
    description="List all conversations.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
):
    """List all conversations."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    convs = conversation_manager.list(limit)
    return {"object": "list", "data": convs, "has_more": False}


@router.get(
    "/conversations/{conversation_id}",
    summary="Get Conversation",
    description="Get a conversation by ID.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    conv = conversation_manager.get(conversation_id)
    if conv is None:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")
    return conv.to_dict()


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete Conversation",
    description="Delete a conversation.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    deleted = conversation_manager.delete(conversation_id)
    if not deleted:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")
    return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}


@router.get(
    "/conversations/{conversation_id}/items",
    summary="Get Conversation Items",
    description="Get items (messages) from a conversation.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def get_conversation_items(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=1000),
    order: str = Query("asc", pattern="^(asc|desc)$"),
):
    """Get items from a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    conv = conversation_manager.get(conversation_id)
    if conv is None:
        raise HTTPException(404, f"Conversation not found: {conversation_id}")

    items = conversation_manager.get_items(conversation_id, limit, order)
    return {"object": "list", "data": items, "has_more": False}


@router.post(
    "/conversations/{conversation_id}/items",
    summary="Add Conversation Items",
    description="Add items (messages) to a conversation.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.CONVERSATION_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.CONVERSATION_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def add_conversation_items(
    conversation_id: str,
    request: Request,
):
    """Add items to a conversation."""
    if conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation manager not initialized.",
        )

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON body: {e}") from e

    items = body.get("items", [])
    if not items:
        raise HTTPException(400, "No items provided")

    try:
        added = conversation_manager.add_items(conversation_id, items)
        return {"object": "list", "data": added}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
