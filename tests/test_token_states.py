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
