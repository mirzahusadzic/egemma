from src.util.file_util import is_likely_binary


def test_a_short_coloured_log_is_text():
    line = b"\x1b[31mERROR\x1b[0m \x1b[1mdisk full\x1b[0m\n"
    assert not is_likely_binary(line * 3)


def test_binary_is_still_binary():
    assert is_likely_binary(bytes(range(256)) * 4)
    assert is_likely_binary(b"\x01\x02\x03\x04" * 50)
