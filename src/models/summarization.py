import logging
import os
import re
from functools import lru_cache

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from ..config import settings

logger = logging.getLogger(__name__)


def _get_persona_system_message(
    persona_name: str, persona_type: str, max_tokens: int, language: str
) -> str:
    """Loads a persona system message from a Markdown file."""
    # Sanitize inputs to prevent path traversal
    sanitized_persona_name = re.sub(r"[^a-zA-Z0-9_]", "", persona_name.lower())
    sanitized_persona_type = re.sub(r"[^a-zA-Z0-9_]", "", persona_type)

    persona_path = os.path.join(
        "personas", sanitized_persona_type, f"{sanitized_persona_name}.md"
    )
    if not os.path.exists(persona_path):
        logger.info(
            f"Persona file not found for {sanitized_persona_name} in "
            f"{sanitized_persona_type}. Falling back to default persona."
        )
        # Fallback to a default persona if a specific one doesn't exist
        persona_path = os.path.join("personas", sanitized_persona_type, "default.md")
        if not os.path.exists(persona_path):
            raise FileNotFoundError(
                f"No persona file found for {persona_name} or default in {persona_type}"
            )
    logger.info(f"Using persona file: {persona_path}")

    with open(persona_path, "r") as f:
        persona_content = f.read()

    # Safely format, only if placeholders exist
    format_kwargs = {"language": language, "max_tokens": max_tokens}
    if "{language}" in persona_content or "{max_tokens}" in persona_content:
        return persona_content.format(**format_kwargs)
    return persona_content


class SummarizationModelWrapper:
    def __init__(self):
        self.model: Llama | None = None

    def load_model(self):
        """Load the local Gemma model for summarization."""
        model_path = hf_hub_download(
            repo_id=settings.SUMMARY_MODEL_NAME,
            filename=settings.SUMMARY_MODEL_BASENAME,
            revision=settings.SUMMARY_MODEL_REVISION,
        )
        self.model = Llama(
            model_path=model_path,
            n_ctx=settings.SUMMARY_N_CTX,
            n_gpu_layers=-1,
            # Gemma's sliding-window layers only attend to their window; without
            # this llama.cpp sizes their cache to the full context.
            swa_full=False,
        )
        logger.info(f"Summarization model loaded: {settings.SUMMARY_MODEL_NAME}")

    def summarize(
        self,
        content: str,
        language: str = "code",
        persona_name: str = "developer",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Summarization model not loaded")
        return self._summarize(content, language, persona_name, max_tokens, temperature)

    @lru_cache(maxsize=128)  # noqa: B019
    def _summarize(
        self,
        content: str,
        language: str,
        persona_name: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        final_max_tokens = (
            max_tokens if max_tokens is not None else settings.SUMMARY_MAX_TOKEN
        )
        final_temperature = (
            temperature if temperature is not None else settings.SUMMARY_TEMP
        )

        persona_type = "docs" if language.lower() == "markdown" else "code"
        system_msg = _get_persona_system_message(
            persona_name, persona_type, final_max_tokens, language
        )

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"Here is the {language} content:\n\n{content}",
            },
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
        )

        return response["choices"][0]["message"]["content"]
