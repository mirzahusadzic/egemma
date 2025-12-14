import pytest

from src.models.summarization import _get_persona_system_message


@pytest.fixture(scope="function")
def setup_test_personas(tmp_path, monkeypatch):
    # Create dummy persona files for testing
    (tmp_path / "personas").mkdir()
    (tmp_path / "personas" / "code").mkdir()
    (tmp_path / "personas" / "docs").mkdir()

    # Default code persona
    (tmp_path / "personas" / "code" / "default.md").write_text("Default code persona")

    # English developer persona
    (tmp_path / "personas" / "code" / "developer.md").write_text(
        "Developer persona for {language}"
    )

    # Default docs persona
    (tmp_path / "personas" / "docs" / "default.md").write_text("Default docs persona")

    # Change to the temp directory to test relative paths
    monkeypatch.chdir(tmp_path)


def test_get_persona_system_message_loads_specific_persona(setup_test_personas):
    """Tests that the correct persona is loaded when a specific persona is requested."""
    message = _get_persona_system_message(
        persona_name="developer", persona_type="code", max_tokens=100, language="python"
    )
    assert message == "Developer persona for python"


def test_get_persona_system_message_falls_back_to_default(setup_test_personas):
    """Tests that the default persona is loaded when a specific persona is not found."""
    message = _get_persona_system_message(
        persona_name="non_existent",
        persona_type="code",
        max_tokens=100,
        language="python",
    )
    assert message == "Default code persona"


def test_get_persona_system_message_raises_error_if_no_persona_found(
    setup_test_personas,
):
    """Tests that a FileNotFoundError is raised if no persona is found."""
    with pytest.raises(FileNotFoundError):
        _get_persona_system_message(
            persona_name="non_existent",
            persona_type="non_existent",
            max_tokens=100,
            language="python",
        )
