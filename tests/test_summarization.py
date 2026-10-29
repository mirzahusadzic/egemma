import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.models.summarization import (
    SummarizationModelWrapper,
    Summary,
    SummaryInputTooLong,
)


def _loaded(content="A summary.", finish_reason="stop", prompt_tokens=10, n_ctx=4096):
    wrapper = SummarizationModelWrapper()
    wrapper.model = MagicMock()
    wrapper.model.tokenize.return_value = [1] * prompt_tokens
    wrapper.model.n_ctx.return_value = n_ctx
    wrapper.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }
    return wrapper


def test_summarization_wrapper_init():
    """Test SummarizationModelWrapper initialization."""
    wrapper = SummarizationModelWrapper()
    assert wrapper.model is None


def test_load_model_uses_the_pinned_checkpoint():
    """The GGUF is fetched at its pinned revision and loaded with a windowed cache."""
    wrapper = SummarizationModelWrapper()

    with (
        patch(
            "src.models.summarization.hf_hub_download", return_value="/m.gguf"
        ) as mock_download,
        patch("src.models.summarization.Llama") as mock_llama,
    ):
        wrapper.load_model()

    mock_download.assert_called_once_with(
        repo_id=settings.SUMMARY_MODEL_NAME,
        filename=settings.SUMMARY_MODEL_BASENAME,
        revision=settings.SUMMARY_MODEL_REVISION,
    )
    kwargs = mock_llama.call_args.kwargs
    assert kwargs["model_path"] == "/m.gguf"
    assert kwargs["n_ctx"] == settings.SUMMARY_N_CTX
    assert kwargs["swa_full"] is False
    assert wrapper.model is mock_llama.return_value


def test_summarize_without_model_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        SummarizationModelWrapper().summarize("def f(): pass", language="Python")


def test_summarize_sends_persona_and_content():
    wrapper = _loaded()

    result = wrapper.summarize(
        "def f(): pass", language="Python", persona_name="developer", max_tokens=64
    )

    assert result == Summary("A summary.", truncated=False)
    kwargs = wrapper.model.create_chat_completion.call_args.kwargs
    system, user = kwargs["messages"]
    assert system["role"] == "system" and system["content"]
    assert user == {
        "role": "user",
        "content": "Here is the Python content:\n\ndef f(): pass",
    }
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == settings.SUMMARY_TEMP


def test_summary_cut_by_max_tokens_is_reported():
    wrapper = _loaded(content="It does th", finish_reason="length")
    assert wrapper.summarize("x = 1", language="Python").truncated


def test_input_that_leaves_no_room_for_the_summary_is_refused():
    wrapper = _loaded(prompt_tokens=2000, n_ctx=4096)
    with pytest.raises(SummaryInputTooLong, match="context holds 4096"):
        wrapper.summarize("x = 1", language="Python", max_tokens=300)
    wrapper.model.create_chat_completion.assert_not_called()


def test_concurrent_summaries_run_one_at_a_time():
    """llama.cpp's context is not thread-safe; calls must not overlap."""
    wrapper = _loaded()
    active, most = 0, 0
    guard = threading.Lock()

    def completion(**kwargs):
        nonlocal active, most
        with guard:
            active += 1
            most = max(most, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return {"choices": [{"message": {"content": "s"}, "finish_reason": "stop"}]}

    wrapper.model.create_chat_completion.side_effect = completion
    threads = [
        threading.Thread(target=wrapper.summarize, args=(f"x = {k}",), daemon=True)
        for k in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert wrapper.model.create_chat_completion.call_count == 4
    assert most == 1
