"""Tests for Harmony format sanitization utilities."""

from src.util.harmony import sanitize_for_display


class TestSanitizeForDisplay:
    """Test sanitize_for_display function."""

    def test_empty_string(self):
        """Test with empty string."""
        assert sanitize_for_display("") == ""

    def test_clean_text_unchanged(self):
        """Test that clean text without Harmony syntax is unchanged."""
        text = "This is a normal thinking block without any Harmony syntax."
        assert sanitize_for_display(text) == text

    def test_remove_assistantcommentary_bash(self):
        """Test removal of assistantcommentary to=functions.bash directive."""
        text = (
            "We need to run commands. Let's use bash.assistantcommentary "
            "to=functions.bash<|constrain|>json"
            '{"command":"cognition-cli patterns list","timeout": 10000}'
        )
        expected = "We need to run commands. Let's use bash."
        assert sanitize_for_display(text) == expected

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
        assert sanitize_for_display(text) == expected

    def test_remove_channel_commentary(self):
        """Test removal of <|channel|>commentary markers."""
        text = "We need to check<|channel|>commentary the status."
        expected = "We need to check the status."
        assert sanitize_for_display(text) == expected

    def test_remove_constrain_tokens(self):
        """Test removal of <|constrain|> tokens."""
        text = "Let's proceed<|constrain|> with the analysis."
        expected = "Let's proceed with the analysis."
        assert sanitize_for_display(text) == expected

    def test_remove_end_tokens(self):
        """Test removal of <|end|> tokens."""
        text = "Analysis complete<|end|> here."
        expected = "Analysis complete here."
        assert sanitize_for_display(text) == expected

    def test_remove_call_tokens(self):
        """Test removal of <|call|> tokens."""
        text = "Making the call<|call|> now."
        expected = "Making the call now."
        assert sanitize_for_display(text) == expected

    def test_remove_multiple_harmony_elements(self):
        """Test removal of multiple Harmony elements in one string."""
        text = (
            "First step<|channel|>commentary then second step<|end|> "
            "finally bash.assistantcommentary to=functions.bash"
            '<|constrain|>json{"cmd":"ls"}'
        )
        expected = "First step then second step finally bash."
        assert sanitize_for_display(text) == expected

    def test_preserve_whitespace_after_removal(self):
        """Test that whitespace is preserved after Harmony removal."""
        text = "Text    before<|channel|>commentary    after"
        result = sanitize_for_display(text)
        # Whitespace should be preserved exactly
        assert result == "Text    before    after"

    def test_preserve_trailing_whitespace(self):
        """Test that trailing whitespace is preserved."""
        text = "Some thinking<|end|>  "
        assert sanitize_for_display(text) == "Some thinking  "

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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
        assert "assistantcommentary" not in result
        assert "<|constrain|>" not in result
        assert "We need architecture overview" in result
        assert "Let's run blast radius for a candidate like ChatModelWrapper." in result

    def test_preserve_json_in_actual_thinking(self):
        """Test that JSON in legitimate thinking (not Harmony) is preserved."""
        text = 'I should create a JSON object like {"key": "value"} for the response.'
        # This should be preserved because it's not a Harmony directive
        assert '{"key": "value"}' in sanitize_for_display(text)

    def test_only_remove_harmony_json(self):
        """Test that only Harmony-formatted JSON is removed."""
        # Harmony directive JSON (should be removed)
        harmony_text = (
            "Run command.assistantcommentary to=functions.bash"
            '<|constrain|>json{"cmd":"ls"}'
        )
        harmony_result = sanitize_for_display(harmony_text)
        assert '{"cmd":"ls"}' not in harmony_result

        # Non-Harmony JSON (should be preserved)
        normal_text = 'The response should be {"status": "ok"} in JSON format.'
        normal_result = sanitize_for_display(normal_text)
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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
        assert "assistantcommentary" not in result
        assert "grepcommentary" not in result
        assert '{"pattern"' not in result
        assert "Let's search the repo." in result
        assert "Use grep tool with search_path." in result

    def test_remove_malformed_to_directive_with_filename(self):
        """Test removal of malformed to= directives with filenames."""
        # From session-1765894201597-openai-magic.log line 86
        text = "We should check changelog to=CHANGELOG.md???? Let's read file."
        result = sanitize_for_display(text)
        assert "to=CHANGELOG.md????" not in result
        assert "We should check changelog" in result
        assert "Let's read file." in result

    def test_remove_start_token(self):
        """Test removal of <|start|> tokens."""
        # From session-1765894201597-openai-magic.log line 86
        text = "Analysis starting<|start|>assistant here we go."
        expected = "Analysis starting here we go."
        assert sanitize_for_display(text) == expected

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
        result = sanitize_for_display(text)
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
        result = sanitize_for_display(text)
        assert result == expected
        assert "assistantanalysis" not in result

    def test_remove_malformed_concatenated_assistant_final(self):
        """Test removal of malformed concatenated assistant+final pattern."""
        text = "Here is the answer.assistantfinal The response is 42."
        # When there's no to=functions directive, we just remove the
        # malformed channel switch but preserve the text after it
        # (which might be legitimate thinking)
        expected = "Here is the answer. The response is 42."
        result = sanitize_for_display(text)
        assert result == expected
        assert "assistantfinal" not in result

    # =========================================================================
    # Pattern 1h tests: Remove standalone JSON tool calls that leak into thinking
    # =========================================================================

    def test_remove_standalone_json_command(self):
        """Test removal of standalone {"command":"..."} JSON in thinking."""
        text = 'Let me count the characters. {"command":"echo -n test | wc -c"} Done.'
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert "Let me count the characters." in result
        assert "Done." in result

    def test_remove_standalone_json_command_with_timeout(self):
        """Test removal of {"command":"...", "timeout": ...} multi-key JSON."""
        text = (
            "Amending commit. "
            '{"command":"git commit --amend --no-edit","timeout": 10000} Done.'
        )
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert '"timeout"' not in result
        assert "Amending commit." in result
        assert "Done." in result

    def test_remove_standalone_json_file_path(self):
        """Test removal of standalone {"file_path":"..."} JSON in thinking."""
        text = 'Reading the config. {"file_path":"/etc/config.json"} Found it.'
        result = sanitize_for_display(text)
        assert '{"file_path"' not in result
        assert "Reading the config." in result
        assert "Found it." in result

    def test_remove_standalone_json_path(self):
        """Test removal of standalone {"path":"..."} JSON in thinking."""
        text = 'Checking directory. {"path":"/usr/local/bin"} Exists.'
        result = sanitize_for_display(text)
        assert '{"path"' not in result
        assert "Checking directory." in result
        assert "Exists." in result

    def test_remove_standalone_json_pattern(self):
        """Test removal of standalone {"pattern":"..."} JSON in thinking."""
        text = 'Searching for errors. {"pattern":"ERROR.*failed"} No matches.'
        result = sanitize_for_display(text)
        assert '{"pattern"' not in result
        assert "Searching for errors." in result
        assert "No matches." in result

    def test_remove_standalone_json_query(self):
        """Test removal of standalone {"query":"..."} JSON in thinking."""
        text = 'Let me search. {"query":"python async await"} Found results.'
        result = sanitize_for_display(text)
        assert '{"query"' not in result
        assert "Let me search." in result
        assert "Found results." in result

    def test_remove_standalone_json_url(self):
        """Test removal of standalone {"url":"..."} JSON in thinking."""
        text = 'Fetching page. {"url":"https://example.com"} Got response.'
        result = sanitize_for_display(text)
        assert '{"url"' not in result
        assert "Fetching page." in result
        assert "Got response." in result

    def test_remove_standalone_json_glob(self):
        """Test removal of standalone {"glob":"..."} JSON in thinking."""
        text = 'Finding files. {"glob":"**/*.py"} Found 50 files.'
        result = sanitize_for_display(text)
        assert '{"glob"' not in result
        assert "Finding files." in result
        assert "Found 50 files." in result

    def test_remove_standalone_json_old_string(self):
        """Test removal of standalone {"old_string":"..."} JSON in thinking."""
        text = 'Editing file. {"old_string":"def foo():"} Replaced.'
        result = sanitize_for_display(text)
        assert '{"old_string"' not in result
        assert "Editing file." in result
        assert "Replaced." in result

    def test_remove_standalone_json_notebook_path(self):
        """Test removal of standalone {"notebook_path":"..."} JSON in thinking."""
        text = 'Opening notebook. {"notebook_path":"analysis.ipynb"} Loaded.'
        result = sanitize_for_display(text)
        assert '{"notebook_path"' not in result
        assert "Opening notebook." in result
        assert "Loaded." in result

    def test_preserve_non_tool_json_in_thinking(self):
        """Test that JSON not matching tool patterns is preserved."""
        # These JSON objects don't match tool call patterns
        text = 'The config is {"name": "test", "enabled": true} and it works.'
        result = sanitize_for_display(text)
        assert '{"name": "test", "enabled": true}' in result

    def test_remove_multiple_standalone_json_tool_calls(self):
        """Test removal of multiple JSON tool calls in one thinking block."""
        text = (
            'First check. {"command":"ls -la"} '
            'Then read. {"file_path":"README.md"} '
            'Finally search. {"pattern":"TODO"} Done.'
        )
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert '{"file_path"' not in result
        assert '{"pattern"' not in result
        assert "First check." in result
        assert "Then read." in result
        assert "Finally search." in result
        assert "Done." in result

    def test_real_world_json_tool_leak(self):
        """Test with real-world example of JSON tool call leaking into thinking."""
        # Based on the actual issue reported
        text = (
            '63 <= 88. Second part: " to serialize concurrent requests '
            "(llama-cpp-python is not thread-safe).\" Let's count. "
            '{"command":"echo -n \' to serialize concurrent requests '
            "(llama-cpp-python is not thread-safe).' | wc -c\"}"
        )
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert "63 <= 88" in result
        assert "Let's count." in result

    def test_json_tool_call_at_end_of_thinking(self):
        """Test JSON tool call at end of thinking block is removed."""
        text = 'I need to run this command {"command":"git status"}'
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert "I need to run this command" in result

    def test_json_tool_call_at_start_of_thinking(self):
        """Test JSON tool call at start of thinking block is removed."""
        text = '{"file_path":"src/main.py"} Let me read this file.'
        result = sanitize_for_display(text)
        assert '{"file_path"' not in result
        assert "Let me read this file." in result

    def test_json_with_nested_quotes(self):
        """Test JSON tool call with nested quotes is removed."""
        text = 'Running: {"command":"echo \\"hello\\""} Complete.'
        result = sanitize_for_display(text)
        assert '{"command"' not in result
        assert "Running:" in result
        assert "Complete." in result

    # =========================================================================
    # Whitespace preservation tests - output must NOT be modified
    # =========================================================================

    def test_preserve_newlines_in_output(self):
        """Test that newlines in clean text are preserved exactly."""
        text = "Line 1\n\nLine 2\n\nLine 3"
        result = sanitize_for_display(text)
        assert result == text, f"Newlines modified: {repr(result)}"

    def test_preserve_multiple_spaces(self):
        """Test that multiple spaces in clean text are preserved."""
        text = "Word1    Word2     Word3"
        result = sanitize_for_display(text)
        assert result == text, f"Spaces modified: {repr(result)}"

    def test_preserve_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is preserved."""
        text = "\n\n  Content here  \n\n"
        result = sanitize_for_display(text)
        assert result == text, f"Whitespace modified: {repr(result)}"

    def test_preserve_markdown_table_formatting(self):
        """Test that markdown table formatting with newlines is preserved."""
        text = """| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |"""
        result = sanitize_for_display(text)
        assert result == text, f"Table formatting modified: {repr(result)}"

    def test_preserve_complex_formatting(self):
        """Test that complex formatting is preserved when no Harmony syntax."""
        text = """## Architecture Overview

| Component | Role |
|-----------|------|
| Wrapper   | LLM  |

### Key Points

1. First point
2. Second point"""
        result = sanitize_for_display(text)
        assert result == text, f"Formatting modified: {repr(result)}"

    def test_only_remove_harmony_keep_everything_else(self):
        """Test that ONLY Harmony syntax is removed, everything else preserved."""
        text = """First line
Second line<|channel|>commentary here
Third line

Fourth line"""
        expected = """First line
Second line here
Third line

Fourth line"""
        result = sanitize_for_display(text)
        assert result == expected, f"Got: {repr(result)}"

    # =========================================================================
    # strip_json parameter tests
    # =========================================================================

    def test_strip_json_true_removes_json(self):
        """Test that strip_json=True (default) removes JSON tool patterns."""
        text = 'Running command {"command":"ls -la"} done.'
        result = sanitize_for_display(text, strip_json=True)
        assert '{"command"' not in result

    def test_strip_json_false_preserves_json(self):
        """Test that strip_json=False preserves JSON in output."""
        text = 'The result is {"command":"ls -la","output":"file.txt"} as shown.'
        result = sanitize_for_display(text, strip_json=False)
        assert '{"command":"ls -la","output":"file.txt"}' in result

    def test_strip_json_false_still_removes_harmony(self):
        """Test that strip_json=False still removes Harmony syntax."""
        text = 'Result<|channel|>commentary is {"data": "value"} here.'
        result = sanitize_for_display(text, strip_json=False)
        assert "<|channel|>" not in result
        assert '{"data": "value"}' in result
