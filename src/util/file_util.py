import string

from ..config import settings


def is_likely_binary(
    content_bytes: bytes, threshold: float = settings.BINARY_DETECTION_THRESHOLD
) -> bool:
    """
    Checks if the given bytes content is likely binary using string.printable.
    A simple heuristic: if a significant portion of bytes are non-printable ASCII
    or null bytes, it's likely binary.
    """
    if not content_bytes:
        return False

    text_chars = set(bytes(string.printable, "ascii"))
    non_text_chars = sum(1 for byte in content_bytes if byte not in text_chars)

    return (non_text_chars / len(content_bytes)) > threshold
