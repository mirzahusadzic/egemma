import re


def condense_log(log_content: str, repetition_threshold: int = 5) -> str:
    """
    Condenses a log file by identifying and collapsing repetitive lines.

    It generalizes lines by replacing timestamps (e.g., `[YYYY-MM-DD HH:MM:SS]`)
    with a `[TIMESTAMP]` placeholder to identify recurring patterns.
    If a generalized line occurs more than `repetition_threshold` times,
    it is condensed into a summary line indicating the repetition count and
    the time range of its occurrences, followed by the generalized line
    with its *first* timestamp restored.

    Args:
        log_content: The full content of the log file as a string.
        repetition_threshold: The minimum number of repetitions required for a
                              line to be condensed. Lines occurring fewer times
                              than this threshold are included as-is.

    Returns:
        A condensed version of the log content as a single string, with
        each line separated by a newline character.
    """
    condensed_lines = []
    # Dictionary to store generalized lines and their original occurrences
    # Key: generalized_line, Value: list of (original_line, timestamp)
    generalized_line_occurrences = {}

    # Regex patterns for common dynamic elements
    TIMESTAMP_PATTERN = re.compile(r"^\[[^]]*\]")

    def generalize_line(line: str) -> tuple[str, str | None]:
        """Replaces the timestamp in a log line with a `[TIMESTAMP]` placeholder
        and extracts the original timestamp.

        Args:
            line: The log line to generalize.

        Returns:
            A tuple containing the generalized line (with `[TIMESTAMP]` placeholder)
            and the extracted original timestamp string,
            or `None` if no timestamp was found.
        """
        original_timestamp = None
        match = TIMESTAMP_PATTERN.match(line)

        if match:
            original_timestamp = match.group(0).strip("[]")
            line = TIMESTAMP_PATTERN.sub("[TIMESTAMP]", line)

        return line, original_timestamp

    for line in log_content.splitlines():
        if not line.strip():
            continue

        generalized_line, timestamp = generalize_line(line)

        if generalized_line not in generalized_line_occurrences:
            generalized_line_occurrences[generalized_line] = []
        generalized_line_occurrences[generalized_line].append((line, timestamp))

    for generalized_line, occurrences in generalized_line_occurrences.items():
        if len(occurrences) >= repetition_threshold:
            first_timestamp = occurrences[0][1]
            last_timestamp = occurrences[-1][1]
            condensed_lines.append(
                f"(Repeated {len(occurrences)} times between "
                f"{first_timestamp} and {last_timestamp})"
            )
            condensed_lines.append(
                generalized_line.replace("[TIMESTAMP]", f"[{first_timestamp}]")
            )  # Restore first timestamp for clarity
        else:
            for original_line, _ in occurrences:
                condensed_lines.append(original_line)

    return "\n".join(condensed_lines)
