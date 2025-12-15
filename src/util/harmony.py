"""
Utility functions for handling Harmony format protocol.

Harmony is GPT-OSS's internal protocol for structured outputs (tool calls, etc.).
These utilities help strip Harmony syntax from user-facing text like thinking blocks.
"""

import re


def sanitize_thinking(thinking: str) -> str:
    """
    Remove Harmony format directives from thinking/reasoning text.

    GPT-OSS models sometimes output Harmony protocol syntax in thinking blocks:
    - assistantcommentary to=functions.bash<|constrain|>json{...}
    - <|channel|>commentary
    - <|constrain|>
    - <|end|>

    This function strips these directives to produce clean thinking text.

    Args:
        thinking: Raw thinking text that may contain Harmony directives

    Returns:
        Cleaned thinking text with Harmony syntax removed

    Examples:
        >>> text = "Let's run bash.assistantcommentary to=functions.bash"
        >>> text += "<|constrain|>json{...}"
        >>> sanitize_thinking(text)
        "Let's run bash."

        >>> sanitize_thinking("We need to check<|channel|>commentary the "
        ...                   "status<|end|>.")
        "We need to check the status."
    """
    if not thinking:
        return thinking

    # Pattern 1: Remove assistantcommentary directives (most common leak)
    # Matches: assistantcommentary to=functions.NAME<|constrain|>json{...}
    # This appears at the end of thinking when tool calls are made
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+<\|constrain\|>json\{.*?\}",
        "",
        thinking,
        flags=re.DOTALL,
    )

    # Pattern 2: Remove Harmony channel markers
    # Matches: <|channel|>commentary
    cleaned = re.sub(r"<\|channel\|>\w+", "", cleaned)

    # Pattern 3: Remove Harmony control tokens
    # Matches: <|constrain|>, <|end|>, <|call|>, etc.
    cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)

    # Pattern 4: Clean up whitespace artifacts from removal
    # Multiple spaces -> single space
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    # Trailing whitespace
    cleaned = cleaned.strip()

    return cleaned
