import re

# A leading timestamp: bracketed ("[2023-01-01 10:00:01] ...") or ISO 8601
# ("2026-01-01T10:00:01.123Z ...", "2026-01-01 10:00:01,123 ..."). A bracket counts
# only if it starts with a digit, so "[INFO]" or "[ERROR]" stays part of the line.
_TIMESTAMP = re.compile(
    r"^(?:\[(?P<bracketed>\d[^]]*)\]"
    r"|(?P<iso>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?))"
)


def condense_log(log_content: str, repetition_threshold: int = 5) -> str:
    """
    Condenses a log file by collapsing lines that repeat apart from their timestamp.

    A line is compared without its leading timestamp, bracketed
    (`[YYYY-MM-DD HH:MM:SS]`) or ISO 8601. A line that occurs at least
    `repetition_threshold` times is replaced, where it first occurs, by a summary
    line giving the count and the time range, followed by its first occurrence; its
    later occurrences are dropped. Every other line keeps its place, so the log
    stays in order. Lines stamped in different styles are counted apart.

    Args:
        log_content: The full content of the log file as a string.
        repetition_threshold: The minimum number of repetitions required for a
                              line to be condensed. Lines occurring fewer times
                              than this threshold are included as-is.

    Returns:
        A condensed version of the log content as a single string, with
        each line separated by a newline character.
    """
    lines: list[tuple[str, tuple[str | None, str]]] = []
    # The timestamps of each line, keyed by the line without its timestamp.
    stamps: dict[tuple[str | None, str], list[str | None]] = {}

    for line in log_content.splitlines():
        if not line.strip():
            continue
        match = _TIMESTAMP.match(line)
        if match:
            kind = match.lastgroup
            key = (kind, line[match.end() :])
            stamp = match.group(kind)
        else:
            key, stamp = (None, line), None
        lines.append((line, key))
        stamps.setdefault(key, []).append(stamp)

    condensed_lines = []
    collapsed = set()
    for line, key in lines:
        occurrences = stamps[key]
        if len(occurrences) < repetition_threshold:
            condensed_lines.append(line)
        elif key not in collapsed:
            collapsed.add(key)
            if occurrences[0] is None:
                condensed_lines.append(f"(Repeated {len(occurrences)} times)")
            else:
                condensed_lines.append(
                    f"(Repeated {len(occurrences)} times between "
                    f"{occurrences[0]} and {occurrences[-1]})"
                )
            condensed_lines.append(line)

    return "\n".join(condensed_lines)
