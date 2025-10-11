import logging
import os
from functools import lru_cache
from typing import Optional

from llama_cpp import Llama

from .config import settings

logger = logging.getLogger(__name__)

SUMMARY_MAX_TOKEN = int(os.getenv("SUMMARY_MAX_TOKEN", 300))
SUMMARY_TEMP = float(os.getenv("SUMMARY_TEMP", 0.2))

# Mapping file extensions to languages/types
EXTENSION_TO_LANGUAGE = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "java": "Java",
    "cpp": "C++",
    "c": "C",
    "go": "Go",
    "rs": "Rust",
    "sh": "Shell script",
    "html": "HTML",
    "css": "CSS",
    "json": "JSON",
    "xml": "XML",
    "sql": "SQL",
    "md": "Markdown",
}


@lru_cache(maxsize=128)
def _cached_summarize(
    model: Llama,
    content: str,
    language: str = "code",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Helper function to cache summarization results."""
    content = content[:8000]  # Trim to context window

    # Determine final token and temperature values
    final_max_tokens = max_tokens if max_tokens is not None else SUMMARY_MAX_TOKEN
    final_temperature = temperature if temperature is not None else SUMMARY_TEMP

    # Use different prompt for Markdown files
    if language.lower() == "markdown":
        system_msg = (
            "You are a helpful assistant that summarizes technical Markdown files "
            "(such as README.md or documentation). Provide a high-level summary, "
            "including project purpose, features, usage, and structure."
        )
    else:
        system_msg = (
            f"You are an expert {language} developer. "
            f"Summarize the following {language} code, explaining its purpose, "
            "main functions, and logic."
        )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": f"Here is the {language} content:\n\n{content}",
        },
    ]

    response = model.create_chat_completion(
        messages=messages,
        temperature=final_temperature,
        max_tokens=final_max_tokens,
    )

    return response["choices"][0]["message"]["content"]


class SummarizationModelWrapper:
    def __init__(self):
        self.model: Optional[Llama] = None

    def load_model(
        self,
    ):
        self.model = Llama.from_pretrained(
            repo_id=settings.SUMMARY_MODEL_NAME,
            filename=settings.SUMMARY_MODEL_BASENAME,
            n_ctx=8192,
            n_gpu_layers=-1,
        )

    def summarize(
        self,
        content: str,
        language: str = "code",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return _cached_summarize(self.model, content, language, max_tokens, temperature)
