"""
Utility functions for handling Harmony format protocol.

Harmony is GPT-OSS's internal protocol for structured outputs (tool calls, etc.).
These utilities help strip Harmony syntax from user-facing text like thinking blocks.
"""

import re


def sanitize_for_display(text: str, strip_json: bool = True) -> str:
    """
    Remove Harmony format directives and malformed tool patterns for display.

    GPT-OSS models sometimes output Harmony protocol syntax and malformed
    tool invocations in thinking blocks and content:
    - assistantcommentary to=functions.bash<|constrain|>json{...}
    - assistantcommentary to=functions.read_file json{...}  (no <|constrain|>)
    - assistantcommentary to=functions.bash code{...}  (code instead of json)
    - assistantcommentary to=functions.grepcommentary{...}  (commentary attached)
    - assistantcommentary to=functions.search_path?commentary
    - to=CHANGELOG.md????
    - <|channel|>commentary
    - <|constrain|>, <|end|>, <|start|>
    - {"command":"..."}, {"file_path":"..."}, etc. (standalone JSON tool calls)

    This function strips these directives to produce clean display text.

    Args:
        text: Raw text that may contain Harmony directives or tool patterns
        strip_json: If True (default), also strip standalone JSON tool calls.
                    Set to False for output text that may contain legitimate JSON
                    (e.g., from tools like `cognition-cli --json`).

    Returns:
        Cleaned text with Harmony syntax removed

    Examples:
        >>> text = "Let's run bash.assistantcommentary to=functions.bash"
        >>> text += "<|constrain|>json{...}"
        >>> sanitize_for_display(text)
        "Let's run bash."

        >>> sanitize_for_display("We need to check<|channel|>commentary the "
        ...                      "status<|end|>.")
        "We need to check the status."
    """
    if not text:
        return text

    # Start with original text
    cleaned = text

    # Pattern 0: Handle malformed concatenation with full directive
    # Matches: [non-whitespace]assistant(commentary|analysis|final)
    # to=functions.X + arguments
    # Example: "file.assistantcommentary to=functions.read_file
    # <|constrain|>json{...}"
    # This is WRONG - model concatenating without proper Harmony delimiters
    # We remove everything from the concatenated "assistant" onwards
    # including the full directive

    # Sub-pattern 0a: With explicit json/code prefix (with optional
    # <|constrain|> and whitespace)
    cleaned = re.sub(
        r"(\S)assistant(commentary|analysis|final)\s+to=functions\.\w+\s*(?:<\|constrain\|>)?\s*(?:json|code)\{.*?\}",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )
    # Sub-pattern 0a2: With JSON but no json/code prefix
    # (e.g., to=functions.grepcommentary{...})
    cleaned = re.sub(
        r"(\S)assistant(commentary|analysis|final)\s+to=functions\.\w+\{.*?\}",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )
    # Sub-pattern 0b: Without JSON payload (e.g., to=functions.X?commentary)
    # Matches optional ?commentary at the end, using word boundary to avoid
    # eating next word
    cleaned = re.sub(
        r"(\S)assistant(commentary|analysis|final)\s+to=functions\.\w+(?:\?commentary)?",
        r"\1",
        cleaned,
    )
    # Sub-pattern 0c: assistant+channel without to=functions
    # (just removes the malformed channel switch)
    cleaned = re.sub(
        r"(\S)assistant(analysis|final)\s+",
        r"\1 ",
        cleaned,
    )

    # Pattern 1a: Remove assistantcommentary directives (most common leak)
    # Matches: assistantcommentary to=functions.NAME<|constrain|>json{...}
    # This appears at the end of thinking when tool calls are made
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+<\|constrain\|>json\{.*?\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Pattern 1b: Remove assistantcommentary without <|constrain|>
    # Matches: assistantcommentary to=functions.NAME json{...}
    # Some tool calls omit the <|constrain|> marker
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+\s+json\{.*?\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Pattern 1c: Remove assistantcommentary with ?commentary suffix
    # Matches: assistantcommentary to=functions.NAME?commentary
    # Appears when model is unsure about tool availability
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+\?commentary",
        "",
        cleaned,
    )

    # Pattern 1d: Remove assistantcommentary with code{...} instead of json{...}
    # Matches: assistantcommentary to=functions.NAME code{...}
    # Some models use 'code' instead of 'json' as the data type marker
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+\s+code\{.*?\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Pattern 1e: Remove assistantcommentary with commentary attached to tool name
    # Matches: assistantcommentary to=functions.NAMEcommentary{...}
    # Example: to=functions.grepcommentary{"pattern":"..."}
    cleaned = re.sub(
        r"assistantcommentary\s+to=functions\.\w+commentary\{.*?\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Pattern 1f: Remove standalone to=functions.NAME code{...}
    # (no assistantcommentary prefix)
    # Matches: to=functions.NAME code{...}
    # Example: to=functions.bash code{"command":"..."}
    cleaned = re.sub(
        r"\s+to=functions\.\w+\s+code\{.*?\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Pattern 1g: Remove malformed to= directives with filenames
    # Matches: to=FILENAME.ext???? or to=PATH/FILE????
    # Example: to=CHANGELOG.md????
    cleaned = re.sub(
        r"\s+to=[\w/.]+\?+",
        "",
        cleaned,
    )

    # Pattern 1h: Remove standalone JSON tool calls that leak into thinking
    # Matches: {"command":"..."}, {"file_path":"..."}, etc.
    # Only applied when strip_json=True (default for thinking blocks)
    if strip_json:
        json_tool_patterns = [
            r'\{"command":[^}]*\}',
            r'\{"file_path":[^}]*\}',
            r'\{"path":[^}]*\}',
            r'\{"pattern":[^}]*\}',
            r'\{"query":[^}]*\}',
            r'\{"url":[^}]*\}',
            r'\{"glob":[^}]*\}',
            r'\{"old_string":[^}]*\}',
            r'\{"notebook_path":[^}]*\}',
        ]
        for pattern in json_tool_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    # Pattern 2: Remove Harmony channel markers
    # Matches: <|channel|>commentary
    cleaned = re.sub(r"<\|channel\|>\w+", "", cleaned)

    # Pattern 3a: Remove <|start|>assistant as a unit
    # Matches: <|start|> followed by 'assistant' keyword
    # Example: <|end|><|start|>assistant -> removes both
    cleaned = re.sub(r"<\|start\|>assistant", "", cleaned)

    # Pattern 3b: Remove other Harmony control tokens
    # Matches: <|constrain|>, <|end|>, <|call|>, etc.
    cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)

    # Pattern 4: Detect remaining malformed concatenation - text directly
    # followed by assistant+channel
    # This runs AFTER other patterns to catch any remaining malformed cases
    # Matches: [non-whitespace]assistant(commentary|analysis|final)
    # Example: After other patterns, "file.assistantcommentary" → "file."
    # This is WRONG - the model is concatenating words without proper Harmony delimiters
    cleaned = re.sub(
        r"(\S)assistant(commentary|analysis|final)(?:\s|$)",
        r"\1 ",
        cleaned,
    )

    # NOTE: We intentionally do NOT modify whitespace.
    # Only Harmony artifacts are removed; all formatting preserved.

    return cleaned
