import logging
from typing import TYPE_CHECKING

import torch

from ..config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _get_optimal_device() -> str:
    if settings.FORCE_CPU:
        logger.info("FORCE_CPU is enabled. Using CPU device.")
        return "cpu"
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA device.")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("MPS is available. Using MPS device.")
        return "mps"
    else:
        logger.info("No GPU acceleration available. Using CPU device.")
        return "cpu"


def span_means(offsets, states, spans):
    """The mean state of the tokens each character span overlaps (a token with
    an empty offset, a special token, never counts); None where none does."""
    means = []
    for start, end in spans:
        rows = [
            k for k, (x, y) in enumerate(offsets) if y > x and x < end and y > start
        ]
        means.append(states[rows].mean(axis=0) if rows else None)
    return means


class TextTooLong(ValueError):
    """A text needs more tokens than the model reads at once."""


class SentenceTransformerWrapper:
    def __init__(self):
        self.model: "SentenceTransformer | None" = None

    def load_model(self):
        import os

        from sentence_transformers import SentenceTransformer

        device = _get_optimal_device()

        # Get HuggingFace token from environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=device,
            token=token,
            revision=settings.EMBEDDING_MODEL_REVISION,
        )

    def encode(
        self, text: str, prompt_name: str | None = None, title: str | None = None
    ):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # EmbeddingGemma's document prompts read "title: none | text: "; a known title
        # takes the place of "none" (it must not add a second header after it).
        if prompt_name in ("document", "Retrieval-document") and title:
            prompt = f"title: {title} | text: "
        else:
            prompt = self.model.prompts[prompt_name] if prompt_name else ""

        # The encoder would cut a longer text silently and embed only its start.
        count = len(self.model.tokenizer(prompt + text)["input_ids"])
        window = self.model.max_seq_length
        if count > window:
            raise TextTooLong(
                f"The text needs {count} tokens with its prompt; the model reads "
                f"{window}. Split it."
            )
        return self.model.encode(text, prompt=prompt)

    def encode_many(self, texts: list[str], prompt_name: str | None = None):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.encode(texts, prompt_name=prompt_name)

    def token_states(self, items: list[tuple[str, list[tuple[int, int]]]]):
        """Each text read as written (no prompt): for each character span asked,
        the mean of the last layer's states of the sub-word tokens it overlaps.
        A span no token overlaps gives None. A text longer than the model's window
        raises TextTooLong before any is read: it is never cut."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        module = self.model[0]
        tokenizer, network = module.tokenizer, module.auto_model
        window = self.model.max_seq_length
        texts = [
            tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
            for text, _ in items
        ]
        for k, encoded in enumerate(texts):
            count = encoded["input_ids"].shape[1]
            if count > window:
                raise TextTooLong(
                    f"Item {k} needs {count} tokens; the model reads {window}. "
                    "Split the text."
                )
        out = []
        with torch.no_grad():
            for encoded, (_, spans) in zip(texts, items, strict=True):
                offsets = encoded.pop("offset_mapping")[0].tolist()
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                states = network(**encoded).last_hidden_state[0].float().cpu().numpy()
                out.append(span_means(offsets, states, spans))
        return out

    @property
    def supported_prompts(self) -> list[str]:
        # These are the prompts listed in the embeddinggemma model card
        return [
            "query",
            "document",
            "BitextMining",
            "Clustering",
            "Classification",
            "InstructionRetrieval",
            "MultilabelClassification",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "Retrieval-query",
            "Retrieval-document",
            "STS",
            "Summarization",
        ]
