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

        # A list of document-style prompts that benefit from having a title
        prompts_with_title_support = ["document", "Retrieval-document"]

        if prompt_name in prompts_with_title_support and title:
            # Format the text with the title for prompts that support it
            formatted_text = f"title: {title} | text: {text}"
        else:
            formatted_text = text

        return self.model.encode(formatted_text, prompt_name=prompt_name)

    def encode_many(self, texts: list[str], prompt_name: str | None = None):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.encode(texts, prompt_name=prompt_name)

    def token_states(self, items: list[tuple[str, list[tuple[int, int]]]]):
        """Each text read as written (no prompt): for each character span asked,
        the mean of the last layer's states of the sub-word tokens it overlaps.
        A span no token overlaps gives None."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        module = self.model[0]
        tokenizer, network = module.tokenizer, module.auto_model
        out = []
        with torch.no_grad():
            for text, spans in items:
                encoded = tokenizer(
                    text, return_offsets_mapping=True, return_tensors="pt"
                )
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
