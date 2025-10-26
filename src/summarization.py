import logging
import os
import re
from functools import lru_cache

import google.generativeai as genai
from llama_cpp import Llama

from .config import settings

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
        self.gemini_client: genai.GenerativeModel | None = None

    def load_model(
        self,
    ):
        self.model = Llama.from_pretrained(
            repo_id=settings.SUMMARY_MODEL_NAME,
            filename=settings.SUMMARY_MODEL_BASENAME,
            n_ctx=settings.SUMMARY_N_CTX,
            n_gpu_layers=-1,
        )
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_client = genai.GenerativeModel(settings.GEMINI_DEFAULT_MODEL)

    def summarize(
        self,
        content: str,
        language: str = "code",
        persona_name: str = "developer",
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_name: str | None = None,
        enable_safety: bool = False,
    ) -> str:
        if model_name and model_name.startswith("gemini"):
            return self._summarize_gemini(
                content,
                language,
                persona_name,
                max_tokens,
                temperature,
                model_name,
                enable_safety,
            )
        elif self.model is None:
            raise RuntimeError("Model not loaded")
        return self._summarize_local(
            content, language, persona_name, max_tokens, temperature
        )

    @lru_cache(maxsize=128)  # noqa: B019
    def _summarize_local(
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

    @lru_cache(maxsize=128)  # noqa: B019
    def _summarize_gemini(
        self,
        content: str,
        language: str,
        persona_name: str,
        max_tokens: int | None,
        temperature: float | None,
        model_name: str | None,
        enable_safety: bool = False,
    ) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

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
            {
                "role": "user",
                "parts": [
                    {"text": system_msg},
                    {"text": f"Here is the {language} content:\n\n{content}"},
                ],
            }
        ]

        model_to_use = model_name if model_name else settings.GEMINI_DEFAULT_MODEL

        # Configure safety settings if enabled
        safety_settings = None
        if enable_safety:
            from google.generativeai.types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
                    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                ),
                HarmCategory.HARM_CATEGORY_HARASSMENT: (
                    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                ),
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
                    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                ),
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
                    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                ),
            }

        response = genai.GenerativeModel(model_to_use).generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                temperature=final_temperature,
                max_output_tokens=final_max_tokens,
            ),
            safety_settings=safety_settings,
            request_options={"timeout": settings.GEMINI_API_TIMEOUT},
        )

        if not response.candidates:
            logger.warning(
                f"Gemini model returned no candidates. Full response: {response}"
            )
            return "No summary could be generated (model returned no candidates)."

        if not response.candidates[0].content.parts:
            logger.warning(
                f"Gemini model returned no content parts. Full response: {response}"
            )
            return "No summary could be generated (model returned no content parts)."

        return response.candidates[0].content.parts[0].text
