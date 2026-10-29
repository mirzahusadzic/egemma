from fastapi import HTTPException, UploadFile, status

from ..config import settings


async def read_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, refused with 413 past MAX_FILE_SIZE_BYTES."""
    limit = settings.MAX_FILE_SIZE_BYTES
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"The file exceeds the {limit:,}-byte limit.",
        )
    return data


def is_likely_binary(
    content_bytes: bytes, threshold: float = settings.BINARY_DETECTION_THRESHOLD
) -> bool:
    """
    Checks if the given bytes content is likely binary.

    Uses a two-stage heuristic:
    1. Try to decode as UTF-8 - if it fails, it's binary
    2. If UTF-8 decoding succeeds, check for null bytes and excessive control characters
       (excluding common ones like newlines, tabs, carriage returns)

    This approach handles UTF-8 text with Unicode characters (e.g., box-drawing, emojis)
    while still detecting actual binary files.
    """
    if not content_bytes:
        return False

    # Try to decode as UTF-8
    try:
        text = content_bytes.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        # If UTF-8 decoding fails, it's likely binary
        return True

    # Check for null bytes (common in binary files, rare in text)
    if "\x00" in text:
        return True

    # Check for excessive control characters (excluding common text control chars)
    # Common text control chars: \n (10), \r (13), \t (9), form feed (12)
    allowed_control_chars = {9, 10, 12, 13}
    control_chars = sum(
        1 for byte in content_bytes if byte < 32 and byte not in allowed_control_chars
    )

    return (control_chars / len(content_bytes)) > threshold
