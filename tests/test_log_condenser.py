from src.util.log_condenser import condense_log


def test_condense_log_no_repetition():
    log_content = (
        "[2023-01-01 10:00:01] INFO: First log message\n"
        "[2023-01-01 10:00:02] DEBUG: Second log message\n"
        "[2023-01-01 10:00:03] ERROR: Third log message"
    )
    expected_output = log_content
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_with_repetition():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Repetitive message\n"
        "[2023-01-01 10:00:02] INFO: Repetitive message\n"
        "[2023-01-01 10:00:03] INFO: Repetitive message\n"
        "[2023-01-01 10:00:04] INFO: Repetitive message\n"
        "[2023-01-01 10:00:05] INFO: Repetitive message"
    )
    expected_output = (
        "(Repeated 5 times between 2023-01-01 10:00:01 and 2023-01-01 10:00:05)\n"
        "[2023-01-01 10:00:01] INFO: Repetitive message"
    )
    assert condense_log(log_content, repetition_threshold=3) == expected_output


def test_condense_log_mixed_content():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Unique message 1\n"
        "[2023-01-01 10:00:02] DEBUG: Repeated message\n"
        "[2023-01-01 10:00:03] DEBUG: Repeated message\n"
        "[2023-01-01 10:00:04] DEBUG: Repeated message\n"
        "[2023-01-01 10:00:05] INFO: Unique message 2"
    )
    expected_output = (
        "[2023-01-01 10:00:01] INFO: Unique message 1\n"
        "(Repeated 3 times between 2023-01-01 10:00:02 and 2023-01-01 10:00:04)\n"
        "[2023-01-01 10:00:02] DEBUG: Repeated message\n"
        "[2023-01-01 10:00:05] INFO: Unique message 2"
    )
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_different_timestamps_same_message():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Message\n"
        "[2023-01-01 10:00:02] INFO: Message\n"
        "[2023-01-01 10:00:03] INFO: Message"
    )
    expected_output = (
        "(Repeated 3 times between 2023-01-01 10:00:01 and 2023-01-01 10:00:03)\n"
        "[2023-01-01 10:00:01] INFO: Message"
    )
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_empty_content():
    log_content = ""
    expected_output = ""
    assert condense_log(log_content) == expected_output


def test_condense_log_threshold_not_met():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Message\n[2023-01-01 10:00:02] INFO: Message"
    )
    expected_output = log_content
    assert condense_log(log_content, repetition_threshold=3) == expected_output


def test_condense_log_no_timestamp():
    log_content = (
        "INFO: Message without timestamp\n"
        "INFO: Message without timestamp\n"
        "INFO: Message without timestamp"
    )
    expected_output = "(Repeated 3 times)\nINFO: Message without timestamp"
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_with_different_dynamic_parts():
    log_content = (
        "[2023-01-01 10:00:01] INFO: User 1 logged in from 192.168.1.1\n"
        "[2023-01-01 10:00:02] INFO: User 2 logged in from 192.168.1.2\n"
        "[2023-01-01 10:00:03] INFO: User 3 logged in from 192.168.1.3"
    )
    expected_output = (
        log_content  # Should not condense as the 'User X' and 'IP' are different
    )
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_with_only_timestamp_change():
    log_content = (
        "[2023-01-01 10:00:01] INFO: System heartbeat\n"
        "[2023-01-01 10:00:02] INFO: System heartbeat\n"
        "[2023-01-01 10:00:03] INFO: System heartbeat\n"
        "[2023-01-01 10:00:04] INFO: System heartbeat\n"
        "[2023-01-01 10:00:05] INFO: System heartbeat"
    )
    expected_output = (
        "(Repeated 5 times between 2023-01-01 10:00:01 and 2023-01-01 10:00:05)\n"
        "[2023-01-01 10:00:01] INFO: System heartbeat"
    )
    assert condense_log(log_content, repetition_threshold=3) == expected_output


def test_condense_log_with_empty_lines():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Message 1\n"
        "\n"
        "[2023-01-01 10:00:02] INFO: Message 2\n"
        "\n"
        "[2023-01-01 10:00:03] INFO: Message 3"
    )
    expected_output = (
        "[2023-01-01 10:00:01] INFO: Message 1\n"
        "[2023-01-01 10:00:02] INFO: Message 2\n"
        "[2023-01-01 10:00:03] INFO: Message 3"
    )
    assert condense_log(log_content, repetition_threshold=2) == expected_output


def test_condense_log_with_multiple_brackets_in_message():
    log_content = (
        "[2023-01-01 10:00:01] INFO: Processed request [ID: 12345] "
        "for user [john.doe]\n"
        "[2023-01-01 10:00:02] INFO: Processed request [ID: 12345] "
        "for user [john.doe]\n"
        "[2023-01-01 10:00:03] INFO: Processed request [ID: 12345] "
        "for user [john.doe]\n"
        "[2023-01-01 10:00:04] INFO: Processed request [ID: 12345] "
        "for user [john.doe]\n"
        "[2023-01-01 10:00:05] INFO: Processed request [ID: 12345] "
        "for user [john.doe]"
    )
    expected_output = (
        "(Repeated 5 times between 2023-01-01 10:00:01 and 2023-01-01 10:00:05)\n"
        "[2023-01-01 10:00:01] INFO: Processed request [ID: 12345] for user [john.doe]"
    )
    assert condense_log(log_content, repetition_threshold=3) == expected_output


def test_condense_log_keeps_the_order_of_lines_it_does_not_collapse():
    log_content = "[t1] start\n[t2] retry\n[t3] connected\n[t4] retry"
    assert condense_log(log_content, repetition_threshold=3) == log_content


def test_condense_log_collapses_iso_timestamped_lines():
    log_content = "\n".join(
        f"2026-01-01T08:00:0{k}.120Z WARN disk slow" for k in range(6)
    )
    assert condense_log(log_content, repetition_threshold=5) == (
        "(Repeated 6 times between 2026-01-01T08:00:00.120Z and "
        "2026-01-01T08:00:05.120Z)\n"
        "2026-01-01T08:00:00.120Z WARN disk slow"
    )


def test_condense_log_collapses_in_place():
    log_content = "\n".join(
        ["2026-01-01 08:00:00,001 INFO boot"]
        + [f"2026-01-01 08:00:0{k},002 WARN retry" for k in range(1, 4)]
        + ["2026-01-01 08:00:05,003 INFO ready"]
    )
    assert condense_log(log_content, repetition_threshold=3).split("\n") == [
        "2026-01-01 08:00:00,001 INFO boot",
        "(Repeated 3 times between 2026-01-01 08:00:01,002 and "
        "2026-01-01 08:00:03,002)",
        "2026-01-01 08:00:01,002 WARN retry",
        "2026-01-01 08:00:05,003 INFO ready",
    ]


def test_condense_log_keeps_levels_apart():
    """[INFO] and [ERROR] are not timestamps: an ERROR is never folded into INFOs."""
    log_content = "\n".join(["[INFO] disk full"] * 5 + ["[ERROR] disk full"])
    assert condense_log(log_content, repetition_threshold=5).split("\n") == [
        "(Repeated 5 times)",
        "[INFO] disk full",
        "[ERROR] disk full",
    ]
