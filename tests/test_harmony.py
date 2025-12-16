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

    def test_remove_assistantcommentary_without_constrain(self):
        """Test removal of assistantcommentary without <|constrain|> marker."""
        # From session-1765893353382-openai-magic.log line 156
        text = (
            "Let's read README maybe includes release notes."
            "assistantcommentary to=functions.read_file "
            'json{"file_path":"README.md","offset":0,"limit":400}'
        )
        expected = "Let's read README maybe includes release notes."
        result = sanitize_thinking(text)
        assert result == expected
        assert "assistantcommentary" not in result
        assert "read_file" not in result

    def test_remove_assistantcommentary_with_question_commentary(self):
        """Test removal of assistantcommentary with ?commentary suffix."""
        # From session-1765893353382-openai-magic.log line 101
        text = (
            "Let's check release notes maybe in docs or README? "
            "Search for v0.5.5 in repo."
            "assistantcommentary to=functions.search_path?commentary"
            "I realize there's no search_path tool; use grep."
        )
        result = sanitize_thinking(text)
        assert "assistantcommentary" not in result
        assert "search_path?commentary" not in result
        assert "Let's check release notes" in result
        assert "I realize there's no search_path tool; use grep." in result

    def test_real_world_mixed_harmony_formats(self):
        """Test with multiple Harmony formats in one text (from real logs)."""
        # Combination of different Harmony formats
        text = (
            "First try search.assistantcommentary to=functions.search_path?commentary"
            "Then use grep.assistantcommentary to=functions.grep"
            '<|constrain|>json{"pattern":"v0.5.5"}'
            "Finally read file.assistantcommentary to=functions.read_file "
            'json{"file_path":"README.md"}'
        )
        result = sanitize_thinking(text)
        # All Harmony syntax should be removed
        assert "assistantcommentary" not in result
        assert "?commentary" not in result
        assert "<|constrain|>" not in result
        assert '{"pattern"' not in result
        assert '{"file_path"' not in result
        # Clean thinking should remain
        assert "First try search." in result
        assert "Then use grep." in result
        assert "Finally read file." in result

    def test_remove_assistantcommentary_with_code_format(self):
        """Test removal of assistantcommentary with code{...} instead of json{...}."""
        # From session-1765894201597-openai-magic.log line 56
        text = (
            "Let's check git tags. "
            "to=functions.bash code"
            '{"command":"git tag --list | grep \\"0.5.5\\""}'
        )
        result = sanitize_thinking(text)
        assert "to=functions.bash" not in result
        assert "code{" not in result
        assert '{"command"' not in result
        assert "Let's check git tags." in result

    def test_remove_assistantcommentary_with_attached_commentary(self):
        """Test removal of assistantcommentary with commentary attached to tool name."""
        # From session-1765894201597-openai-magic.log line 111
        text = (
            "Let's search the repo. Use grep tool with search_path."
            "assistantcommentary to=functions.grepcommentary"
            '{"pattern":"0.5.5","search_path":"."}'
        )
        result = sanitize_thinking(text)
        assert "assistantcommentary" not in result
        assert "grepcommentary" not in result
        assert '{"pattern"' not in result
        assert "Let's search the repo." in result
        assert "Use grep tool with search_path." in result

    def test_remove_malformed_to_directive_with_filename(self):
        """Test removal of malformed to= directives with filenames."""
        # From session-1765894201597-openai-magic.log line 86
        text = "We should check changelog to=CHANGELOG.md???? Let's read file."
        result = sanitize_thinking(text)
        assert "to=CHANGELOG.md????" not in result
        assert "We should check changelog" in result
        assert "Let's read file." in result

    def test_remove_start_token(self):
        """Test removal of <|start|> tokens."""
        # From session-1765894201597-openai-magic.log line 86
        text = "Analysis starting<|start|>assistant here we go."
        expected = "Analysis starting here we go."
        assert sanitize_thinking(text) == expected

    def test_remove_malformed_concatenated_assistant_commentary(self):
        """Test removal of malformed concatenated assistant+channel pattern."""
        # When model outputs text directly followed by "assistant" + channel name
        # without proper Harmony delimiters. This is wrong - it's mixing channels.
        # Example: "file.assistantcommentary to=functions.read_file"
        text = (
            "Let's confirm. Open the test "
            "file.assistantcommentary to=functions.read_file "
            '<|constrain|>json{"file_path":"tests/test_harmony_format.py", '
            '"offset":70, "limit":60}'
        )
        expected = "Let's confirm. Open the test file."
        result = sanitize_thinking(text)
        assert result == expected
        assert "assistantcommentary" not in result
        assert "read_file" not in result

    def test_remove_malformed_concatenated_assistant_analysis(self):
        """Test removal of malformed concatenated assistant+analysis pattern."""
        text = "Let me think about this.assistantanalysis This is wrong formatting."
        # When there's no to=functions directive, we just remove the
        # malformed channel switch but preserve the text after it
        # (which might be legitimate thinking)
        expected = "Let me think about this. This is wrong formatting."
        result = sanitize_thinking(text)
        assert result == expected
        assert "assistantanalysis" not in result

    def test_remove_malformed_concatenated_assistant_final(self):
        """Test removal of malformed concatenated assistant+final pattern."""
        text = "Here is the answer.assistantfinal The response is 42."
        # When there's no to=functions directive, we just remove the
        # malformed channel switch but preserve the text after it
        # (which might be legitimate thinking)
        expected = "Here is the answer. The response is 42."
        result = sanitize_thinking(text)
        assert result == expected
        assert "assistantfinal" not in result
