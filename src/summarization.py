import logging
import os
from functools import lru_cache
from typing import Optional

from llama_cpp import Llama

from .config import settings

logger = logging.getLogger(__name__)


def _get_persona_system_message(
    language: str, persona_type: str, max_tokens: int
) -> str:
    """Loads a persona system message from a Markdown file."""
    persona_path = os.path.join("personas", persona_type, f"{language.lower()}.md")
    if not os.path.exists(persona_path):
        # Fallback to a default persona if a specific one doesn't exist
        persona_path = os.path.join("personas", persona_type, "default.md")
        if not os.path.exists(persona_path):
            raise FileNotFoundError(
                f"No persona file found for {language} or default in {persona_type}"
            )

    with open(persona_path, "r") as f:
        persona_content = f.read()

    # Safely format, only if placeholders exist
    format_kwargs = {"language": language, "max_tokens": max_tokens}
    if "{language}" in persona_content or "{max_tokens}" in persona_content:
        return persona_content.format(**format_kwargs)
    return persona_content


@lru_cache(maxsize=128)
def _cached_summarize(
    model: Llama,
    content: str,
    language: str = "code",
    persona_name: str = "developer",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Helper function to cache summarization results."""
    content = content[:8000]  # Trim to context window

    # Determine final token and temperature values
    final_max_tokens = (
        max_tokens if max_tokens is not None else settings.SUMMARY_MAX_TOKEN
    )
    final_temperature = (
        temperature if temperature is not None else settings.SUMMARY_TEMP
    )

    # Determine persona type based on language
    persona_type = "docs" if language.lower() == "markdown" else "code"
    system_msg = _get_persona_system_message(
        persona_name, persona_type, final_max_tokens
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
        persona_name: str = "developer",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return _cached_summarize(
            self.model, content, language, persona_name, max_tokens, temperature
        )
