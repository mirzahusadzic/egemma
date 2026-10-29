import numpy as np

from src.models.embedding import span_means


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
