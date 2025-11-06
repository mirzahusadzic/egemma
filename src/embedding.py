import logging
from typing import TYPE_CHECKING

import torch

from src.config import settings

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
        self.model: "SentenceTransformer | None" = None

    def load_model(self):
        import os

        from sentence_transformers import SentenceTransformer

        device = _get_optimal_device()

        # Get HuggingFace token from environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME, device=device, token=token
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
