import logging
import os
import re
from functools import lru_cache

from google import genai
from google.genai import types
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
        self.gemini_client: genai.Client | None = None

    def load_local_model(self):
        """Load the local Llama model for summarization."""
        self.model = Llama.from_pretrained(
            repo_id=settings.SUMMARY_MODEL_NAME,
            filename=settings.SUMMARY_MODEL_BASENAME,
            n_ctx=settings.SUMMARY_N_CTX,
            n_gpu_layers=-1,
        )
        logger.info(f"Local summarization model loaded: {settings.SUMMARY_MODEL_NAME}")

    def load_gemini_client(self):
        """Initialize the Gemini API client."""
        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini API client initialized")
        else:
            logger.warning("GEMINI_API_KEY not set, Gemini client not initialized")

    def load_model(self):
        """Load both local model and Gemini client.

        Deprecated: use load_local_model and load_gemini_client instead.
        """
        self.load_local_model()
        self.load_gemini_client()

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
        # If a Gemini model is explicitly requested, use Gemini
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
        # Otherwise, use the local model
        elif self.model is None:
            raise RuntimeError(
                "Local summarization model not loaded. "
                "Either enable SUMMARY_LOCAL_ENABLED or specify a Gemini model."
            )
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

        model_to_use = model_name if model_name else settings.GEMINI_DEFAULT_MODEL

        # Configure safety settings if enabled
        safety_settings = None
        if enable_safety:
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
            ]

        # Build config object
        config = types.GenerateContentConfig(
            temperature=final_temperature,
            max_output_tokens=final_max_tokens,
            system_instruction=system_msg,
            safety_settings=safety_settings,
        )

        # Generate content using new SDK with error handling
        try:
            response = self.gemini_client.models.generate_content(
                model=model_to_use,
                contents=f"Here is the {language} content:\n\n{content}",
                config=config,
            )
        except Exception as e:
            from google.genai import errors

            # Handle specific Gemini API errors
            if isinstance(e, errors.ClientError):
                status_code = e.status_code
                error_msg = str(e)

                if status_code == 429:
                    logger.warning(f"Gemini API quota exceeded: {error_msg}")
                    return (
                        "API quota exceeded. Please try again later or "
                        "reduce request frequency. Check "
                        "https://ai.google.dev/gemini-api/docs/rate-limits"
                    )
                elif status_code == 503:
                    logger.warning(f"Gemini API unavailable: {error_msg}")
                    return (
                        "Gemini API is temporarily overloaded. "
                        "Please try again in a few moments."
                    )
                else:
                    logger.error(f"Gemini API error {status_code}: {error_msg}")
                    return (
                        f"API error ({status_code}): Unable to generate "
                        "summary. Please try again."
                    )
            else:
                # Unexpected error
                logger.error(f"Unexpected error calling Gemini API: {e}")
                return "An unexpected error occurred while generating the summary."

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
