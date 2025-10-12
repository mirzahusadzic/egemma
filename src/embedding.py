import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _get_optimal_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA device.")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("MPS is available. Using MPS device.")
        return "mps"
    else:
        logger.info("No GPU acceleration available. Using CPU device.")
        return "cpu"


class SentenceTransformerWrapper:
    def __init__(self):
        self.model: "Optional[SentenceTransformer]" = None

    def load_model(self):
        from sentence_transformers import SentenceTransformer

        device = _get_optimal_device()
        self.model = SentenceTransformer("google/embeddinggemma-300m", device=device)

    def encode(
        self, text: str, prompt_name: Optional[str] = None, title: Optional[str] = None
    ):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if prompt_name == "document" and title:
            # Format the text with the title for the 'document' prompt
            formatted_text = f"title: {title} | text: {text}"
        else:
            formatted_text = text

        return self.model.encode(formatted_text, prompt_name=prompt_name)

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
