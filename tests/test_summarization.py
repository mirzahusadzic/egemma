from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.models.summarization import SummarizationModelWrapper


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
    wrapper = SummarizationModelWrapper()
    wrapper.model = MagicMock()
    wrapper.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A summary."}}]
    }

    result = wrapper.summarize(
        "def f(): pass", language="Python", persona_name="developer", max_tokens=64
    )

    assert result == "A summary."
    kwargs = wrapper.model.create_chat_completion.call_args.kwargs
    system, user = kwargs["messages"]
    assert system["role"] == "system" and system["content"]
    assert user == {
        "role": "user",
        "content": "Here is the Python content:\n\ndef f(): pass",
    }
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == settings.SUMMARY_TEMP
