import logging
import os
import re
import threading
from typing import NamedTuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from ..config import settings
from ..util.content_cache import ContentCache

logger = logging.getLogger(__name__)

_PERSONA_FIELD = re.compile(r"\{\{|\}\}|\{(language|max_tokens)\}")

# The chat template's own tokens around the two turns, with room to spare.
_TEMPLATE_TOKENS = 64


class SummaryInputTooLong(ValueError):
    """The file and its persona leave the summary no room in the context."""


class Summary(NamedTuple):
    text: str
    truncated: bool  # max_tokens ended the summary, not the model


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

    # Fill the two placeholders. Doubled braces stand for one, as in str.format;
    # any other brace (a JSON example, say) is text.
    values = {"language": language, "max_tokens": str(max_tokens)}
    return _PERSONA_FIELD.sub(
        lambda m: values[m.group(1)] if m.group(1) else m.group(0)[0],
        persona_content,
    )


class SummarizationModelWrapper:
    def __init__(self):
        self.model: Llama | None = None
        # One llama.cpp context: concurrent calls would share and corrupt it.
        self._inference_lock = threading.Lock()
        # Summaries by a hash of the prompt that produced them, persona text included.
        self.cache = ContentCache(settings.CACHE_MAX_SIZE)

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
            # Otherwise llama.cpp prints the GGUF metadata and chat template on load.
            verbose=False,
        )
        logger.info(f"Summarization model loaded: {settings.SUMMARY_MODEL_NAME}")

    def summarize(
        self,
        content: str,
        language: str = "code",
        persona_name: str = "developer",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Summary:
        if self.model is None:
            raise RuntimeError("Summarization model not loaded")

        final_max_tokens = (
            max_tokens if max_tokens is not None else settings.SUMMARY_MAX_TOKEN
        )
        final_temperature = (
            temperature if temperature is not None else settings.SUMMARY_TEMP
        )

        # Read on every call, so an edited persona takes effect at once.
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
        key = ContentCache.key(messages, final_max_tokens, final_temperature)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        with self._inference_lock:
            # An identical request may have been answered while this one waited.
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            prompt_tokens = _TEMPLATE_TOKENS + sum(
                len(self.model.tokenize(m["content"].encode("utf-8"), add_bos=False))
                for m in messages
            )
            context = self.model.n_ctx()
            if prompt_tokens + final_max_tokens > context:
                raise SummaryInputTooLong(
                    f"The file and its persona need about {prompt_tokens} tokens and "
                    f"the summary up to {final_max_tokens}; the context holds "
                    f"{context}. Split the file, or raise SUMMARY_N_CTX."
                )
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            )
            choice = response["choices"][0]
            summary = Summary(
                choice["message"]["content"], choice.get("finish_reason") == "length"
            )
            self.cache.put(key, summary)
        return summary
