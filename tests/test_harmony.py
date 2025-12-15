"""Tests for Harmony format sanitization utilities."""

from src.util.harmony import sanitize_thinking


class TestSanitizeThinking:
    """Test sanitize_thinking function."""

    def test_empty_string(self):
        """Test with empty string."""
        assert sanitize_thinking("") == ""

    def test_clean_text_unchanged(self):
        """Test that clean text without Harmony syntax is unchanged."""
        text = "This is a normal thinking block without any Harmony syntax."
        assert sanitize_thinking(text) == text

    def test_remove_assistantcommentary_bash(self):
        """Test removal of assistantcommentary to=functions.bash directive."""
        text = (
            "We need to run commands. Let's use bash.assistantcommentary "
            "to=functions.bash<|constrain|>json"
            '{"command":"cognition-cli patterns list","timeout": 10000}'
        )
        expected = "We need to run commands. Let's use bash."
        assert sanitize_thinking(text) == expected

    def test_remove_assistantcommentary_multiline(self):
        """Test removal of assistantcommentary with multiline JSON."""
        text = (
            "We should analyze the blast radius.assistantcommentary "
            "to=functions.bash<|constrain|>json{\n"
            '  "command": "cognition-cli blast-radius ChatModelWrapper",\n'
            '  "timeout": 10000\n'
            "}"
        )
        expected = "We should analyze the blast radius."
        assert sanitize_thinking(text) == expected

    def test_remove_channel_commentary(self):
        """Test removal of <|channel|>commentary markers."""
        text = "We need to check<|channel|>commentary the status."
        expected = "We need to check the status."
        assert sanitize_thinking(text) == expected

    def test_remove_constrain_tokens(self):
        """Test removal of <|constrain|> tokens."""
        text = "Let's proceed<|constrain|> with the analysis."
        expected = "Let's proceed with the analysis."
        assert sanitize_thinking(text) == expected

    def test_remove_end_tokens(self):
        """Test removal of <|end|> tokens."""
        text = "Analysis complete<|end|> here."
        expected = "Analysis complete here."
        assert sanitize_thinking(text) == expected

    def test_remove_call_tokens(self):
        """Test removal of <|call|> tokens."""
        text = "Making the call<|call|> now."
        expected = "Making the call now."
        assert sanitize_thinking(text) == expected

    def test_remove_multiple_harmony_elements(self):
        """Test removal of multiple Harmony elements in one string."""
        text = (
            "First step<|channel|>commentary then second step<|end|> "
            "finally bash.assistantcommentary to=functions.bash"
            '<|constrain|>json{"cmd":"ls"}'
        )
        expected = "First step then second step finally bash."
        assert sanitize_thinking(text) == expected

    def test_clean_up_whitespace(self):
        """Test that excessive whitespace is cleaned up after removal."""
        text = "Text    before<|channel|>commentary    after"
        result = sanitize_thinking(text)
        # Should not have multiple consecutive spaces
        assert "  " not in result
        assert result == "Text before after"

    def test_strip_trailing_whitespace(self):
        """Test that trailing whitespace is removed."""
        text = "Some thinking<|end|>  "
        assert sanitize_thinking(text) == "Some thinking"

    def test_real_world_example_from_logs(self):
        """Test with actual example from eGemma logs."""
        # From tui-oss20b-1765755680907-openai-magic.log line 41
        text = (
            "We have huge list, not needed. "
            "We need architecture overview. "
            "Use PGC CLI commands: 'cognition-cli patterns list', "
            "'patterns analyze', etc. "
            "Likely available. Use bash.assistantcommentary "
            "to=functions.bash<|constrain|>json"
            '{"command":"cognition-cli patterns list","timeout": 10000}'
        )
        result = sanitize_thinking(text)
        # Should not contain any Harmony syntax
        assert "assistantcommentary" not in result
        assert "<|constrain|>" not in result
        assert '{"command"' not in result
        # Should contain the actual thinking
        assert "We have huge list, not needed" in result
        assert "Use PGC CLI commands" in result

    def test_real_world_example_blast_radius(self):
        """Test with blast radius example from logs."""
        # From tui-oss20b-1765755680907-openai-magic.log line 61
        text = (
            "We need architecture overview. "
            "We have patterns list and analysis. "
            "We should identify core components, maybe ChatModelWrapper, "
            "ConversationManager, "
            "Response, etc. Use blast-radius to find high impact. "
            "Let's run blast radius for a "
            "candidate like ChatModelWrapper.assistantcommentary "
            "to=functions.bash<|constrain|>json"
            '{"command":"cognition-cli blast-radius \\"ChatModelWrapper\\" '
            '--json","timeout": 10000}'
        )
        result = sanitize_thinking(text)
        assert "assistantcommentary" not in result
        assert "<|constrain|>" not in result
        assert "We need architecture overview" in result
        assert "Let's run blast radius for a candidate like ChatModelWrapper." in result

    def test_preserve_json_in_actual_thinking(self):
        """Test that JSON in legitimate thinking (not Harmony) is preserved."""
        text = 'I should create a JSON object like {"key": "value"} for the response.'
        # This should be preserved because it's not a Harmony directive
        assert '{"key": "value"}' in sanitize_thinking(text)

    def test_only_remove_harmony_json(self):
        """Test that only Harmony-formatted JSON is removed."""
        # Harmony directive JSON (should be removed)
        harmony_text = (
            "Run command.assistantcommentary to=functions.bash"
            '<|constrain|>json{"cmd":"ls"}'
        )
        harmony_result = sanitize_thinking(harmony_text)
        assert '{"cmd":"ls"}' not in harmony_result

        # Non-Harmony JSON (should be preserved)
        normal_text = 'The response should be {"status": "ok"} in JSON format.'
        normal_result = sanitize_thinking(normal_text)
        assert '{"status": "ok"}' in normal_result
