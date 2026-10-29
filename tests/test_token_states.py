from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.models.embedding import SentenceTransformerWrapper, TextTooLong, span_means


def test_span_means_averages_the_tokens_a_span_overlaps():
    offsets = [
        (0, 0),
        (0, 3),
        (4, 8),
        (8, 11),
        (0, 0),
    ]  # special, "The", "with", "held", special
    states = np.array([[9.0, 9.0], [1.0, 1.0], [2.0, 4.0], [4.0, 8.0], [9.0, 9.0]])
    means = span_means(offsets, states, [(4, 11), (0, 3), (20, 25)])
    assert np.allclose(means[0], [3.0, 6.0])  # "withheld": both of its tokens
    assert np.allclose(means[1], [1.0, 1.0])
    assert means[2] is None  # no token there


def test_a_text_past_the_window_is_refused_before_any_is_read():
    def tokenizer(text, **kwargs):
        n = len(text.split()) + 2  # with the boundary tokens
        return {
            "input_ids": torch.ones(1, n, dtype=torch.long),
            "offset_mapping": torch.zeros(1, n, 2, dtype=torch.long),
        }

    module = MagicMock(tokenizer=tokenizer)
    wrapper = SentenceTransformerWrapper()
    wrapper.model = MagicMock(max_seq_length=6)
    wrapper.model.__getitem__.return_value = module

    with pytest.raises(TextTooLong, match="Item 1 needs 7 tokens; the model reads 6"):
        wrapper.token_states([("one two", [(0, 3)]), ("a b c d e", [(0, 1)])])
    module.auto_model.assert_not_called()


def test_a_file_past_the_window_is_refused_not_cut():
    """/embed's encode counts the prompt's tokens too, and never embeds a prefix."""
    wrapper = SentenceTransformerWrapper()
    wrapper.model = MagicMock(
        max_seq_length=8, prompts={"document": "title: none | text: "}
    )
    wrapper.model.tokenizer.side_effect = lambda text: {"input_ids": text.split()}

    with pytest.raises(TextTooLong, match="needs 9 tokens with its prompt"):
        wrapper.encode("a b c d e", prompt_name="document")
    wrapper.model.encode.assert_not_called()

    wrapper.encode("a b c d", prompt_name="document")
    wrapper.model.encode.assert_called_once_with(
        "a b c d", prompt="title: none | text: "
    )


def test_a_title_replaces_none_in_the_document_prompt():
    wrapper = SentenceTransformerWrapper()
    wrapper.model = MagicMock(
        max_seq_length=64, prompts={"document": "title: none | text: "}
    )
    wrapper.model.tokenizer.side_effect = lambda text: {"input_ids": text.split()}

    wrapper.encode("the body", prompt_name="document", title="Pump spec")
    wrapper.model.encode.assert_called_once_with(
        "the body", prompt="title: Pump spec | text: "
    )
