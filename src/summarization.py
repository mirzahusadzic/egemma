import logging
import os
from typing import Optional

from llama_cpp import Llama

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


class SummarizationModelWrapper:
    def __init__(self):
        self.model: Optional[Llama] = None

    def load_model(
        self,
        model_name: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
        model_basename: str = "gemma-3-12b-it-q4_0.gguf",
    ):
        self.model = Llama.from_pretrained(
            repo_id=model_name,
            filename=model_basename,
            n_ctx=8192,
            n_gpu_layers=-1,
        )

    def summarize(self, content: str, language: str = "code") -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        content = content[:8000]  # Trim to context window

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

        response = self.model.create_chat_completion(
            messages=messages,
            temperature=SUMMARY_TEMP,
            max_tokens=SUMMARY_MAX_TOKEN,
        )

        return response["choices"][0]["message"]["content"]
