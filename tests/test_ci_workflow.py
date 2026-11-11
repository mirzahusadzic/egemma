from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def top_level_block(text: str, key: str) -> list[str] | None:
    """The indented lines under a top-level key, or None without the key."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == f"{key}:":
            block = []
            for body in lines[i + 1 :]:
                if body.strip() and not body[0].isspace():
                    break
                if body.strip() and not body.lstrip().startswith("#"):
                    block.append(body.strip())
            return block
    return None


def test_every_workflow_token_only_reads_contents():
    workflows = sorted(WORKFLOWS.glob("*.yml"))
    assert workflows
    for workflow in workflows:
        text = workflow.read_text()
        name = workflow.name
        assert top_level_block(text, "permissions") == ["contents: read"], name
        assert "permissions:" not in text.replace("\npermissions:", "", 1), name
