from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_conversations_dir(model_name: str) -> Path:
    """Get the conversations directory for a model.

    Returns ~/.egemma/{model-name}/conversations
    """
    return Path.home() / ".egemma" / model_name / "conversations"


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    MODEL_PROFILE: Literal["workbench", "semantic"] = "workbench"
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_MODEL_REVISION: str = "57c266a740f537b4dc058e1b0cda161fd15afa75"
    EMBEDDING_BATCH_MAX_INPUTS: int = 256
    EMBEDDING_MAX_INPUT_CHARS: int = 16_384
    AMRBART_MODEL_ENABLED: bool = False
    AMRBART_MODEL_NAME: str = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
    AMRBART_MODEL_REVISION: str = "12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512"
    AMRBART_DEVICE: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    AMRBART_MAX_SOURCE_TOKENS: int = 400
    AMRBART_MAX_GRAPH_TOKENS: int = 1024
    AMRBART_NUM_BEAMS: int = 5
    AMRBART_MAX_INPUT_CHARS: int = 16_384
    AMRBART_BATCH_MAX_INPUTS: int = 64
    AMRBART_RATE_LIMIT_SECONDS: int = 60
    AMRBART_RATE_LIMIT_CALLS: int = 10
    SUMMARY_LOCAL_ENABLED: bool = True
    # Chat Model Settings (GPT-OSS-20B)
    CHAT_MODEL_ENABLED: bool = True
    CHAT_MODEL_PATH: str = "models/gpt-oss-20b-F16.gguf"
    CHAT_MODEL_NAME: str = "gpt-oss-20b"
    CHAT_N_CTX: int = 65536  # Context window (64K - stable on 36GB)
    CHAT_N_GPU_LAYERS: int = -1  # -1 = all layers on GPU (Metal)
    CHAT_N_BATCH: int = 2048  # Batch size for prompt processing
    CHAT_USE_MMAP: bool = False  # Disable mmap to prevent M3 freeze during load
    CHAT_FLASH_ATTN: bool = True  # Enable Flash Attention for Metal
    CHAT_MAX_TOKENS: int = 4096
    CHAT_TEMPERATURE: float = 0.3
    CHAT_MIN_P: float = 0.05  # Prevents degenerate sampling
    CHAT_TOP_P: float = 0.95  # Nucleus sampling
    CHAT_TOP_K: int = 0  # Top-K sampling for diversity
    CHAT_REASONING_EFFORT: str = "high"  # low, medium, high - controls thinking depth
    CHAT_RATE_LIMIT_SECONDS: int = 60
    CHAT_RATE_LIMIT_CALLS: int = 100
    # Conversation API rate limiting (generous limits for stateful operations)
    CONVERSATION_RATE_LIMIT_SECONDS: int = 60
    CONVERSATION_RATE_LIMIT_CALLS: int = 100
    SUMMARY_MODEL_NAME: str = "google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME: str = "gemma-3-12b-it-q4_0.gguf"
    SUMMARY_MAX_TOKEN: int = 300
    SUMMARY_TEMP: float = 0.7
    SUMMARY_N_CTX: int = 8192
    GEMINI_API_KEY: str | None = None
    GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_TIMEOUT: int = 60
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    BINARY_DETECTION_THRESHOLD: float = 0.1
    CACHE_MAX_SIZE: int = 128
    EMBED_RATE_LIMIT_SECONDS: int = 10
    EMBED_RATE_LIMIT_CALLS: int = 5
    SUMMARIZE_RATE_LIMIT_SECONDS: int = 60
    SUMMARIZE_RATE_LIMIT_CALLS: int = 2
    FORCE_CPU: bool = False
    EXTENSION_TO_LANGUAGE: dict[str, str] = {
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
        "yml": "YAML",
        "json": "JSON",
        "xml": "XML",
        "sql": "SQL",
        "md": "Markdown",
        "log": "Log File",
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def amrbart_enabled(self) -> bool:
        """Whether the AMR parser belongs to the selected model profile."""
        return self.MODEL_PROFILE == "semantic" or self.AMRBART_MODEL_ENABLED

    @property
    def chat_enabled(self) -> bool:
        """Whether the selected profile permits the local chat model."""
        return self.MODEL_PROFILE == "workbench" and self.CHAT_MODEL_ENABLED

    @property
    def local_summary_enabled(self) -> bool:
        """Whether the selected profile permits the local summarization model."""
        return self.MODEL_PROFILE == "workbench" and self.SUMMARY_LOCAL_ENABLED

    @property
    def gemini_summary_enabled(self) -> bool:
        """Whether the selected profile permits cloud summarization."""
        return self.MODEL_PROFILE == "workbench" and bool(self.GEMINI_API_KEY)


settings = Settings()
