from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_conversations_dir(model_name: str) -> Path:
    """Get the conversations directory for a model.

    Returns ~/.egemma/{model-name}/conversations
    """
    return Path.home() / ".egemma" / model_name / "conversations"


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    SUMMARY_LOCAL_ENABLED: bool = True
    # Chat Model Settings (GPT-OSS-20B)
    CHAT_MODEL_ENABLED: bool = True
    CHAT_MODEL_PATH: str = "models/gpt-oss-20b-F16.gguf"
    CHAT_MODEL_NAME: str = "gpt-oss-20b"
    CHAT_N_CTX: int = 65536  # Context window (64K - stable on 32GB)
    CHAT_N_GPU_LAYERS: int = -1  # -1 = all layers on GPU (Metal)
    CHAT_N_BATCH: int = 2048  # Batch size for prompt processing
    CHAT_USE_MMAP: bool = False  # Disable mmap to prevent M3 freeze during load
    CHAT_FLASH_ATTN: bool = True  # Enable Flash Attention for Metal
    CHAT_MAX_TOKENS: int = 4096
    CHAT_TEMPERATURE: float = 1.0
    CHAT_MIN_P: float = 0.05  # Prevents degenerate sampling
    CHAT_TOP_P: float = 1.0  # Nucleus sampling (Unsloth recommendation)
    CHAT_TOP_K: int = 40  # Top-K sampling for diversity
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


settings = Settings()
