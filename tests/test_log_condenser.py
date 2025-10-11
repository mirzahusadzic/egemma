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
    expected_output = (
        "(Repeated 3 times between None and None)\nINFO: Message without timestamp"
    )
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
