from pathlib import Path

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


def test_braces_other_than_the_placeholders_are_text(setup_test_personas):
    """A JSON example beside a placeholder no longer breaks the persona."""
    Path("personas/docs/validator.md").write_text(
        'Answer in {max_tokens} tokens as:\n{\n  "score": 0.9\n}\nfor {language}; {{x}}'
    )
    message = _get_persona_system_message(
        persona_name="validator", persona_type="docs", max_tokens=300, language="Go"
    )
    assert message == 'Answer in 300 tokens as:\n{\n  "score": 0.9\n}\nfor Go; {x}'


@pytest.mark.parametrize(
    "path", sorted(Path(__file__).parent.parent.glob("personas/*/*.md")), ids=str
)
def test_every_shipped_persona_renders(path, monkeypatch):
    """Each persona fills; where str.format worked before, the text is unchanged."""
    monkeypatch.chdir(path.parent.parent.parent)
    message = _get_persona_system_message(
        persona_name=path.stem,
        persona_type=path.parent.name,
        max_tokens=300,
        language="Python",
    )
    assert "{language}" not in message and "{max_tokens}" not in message
    try:
        expected = path.read_text().format(language="Python", max_tokens=300)
    except (KeyError, IndexError, ValueError):
        return  # str.format rejected this persona's literal braces
    assert message == expected
