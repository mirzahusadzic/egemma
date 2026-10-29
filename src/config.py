from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_MODEL_REVISION: str = "57c266a740f537b4dc058e1b0cda161fd15afa75"
    EMBEDDING_BATCH_MAX_INPUTS: int = 256
    EMBEDDING_MAX_INPUT_CHARS: int = 16_384
    AMRBART_MODEL_ENABLED: bool = True
    AMRBART_MODEL_NAME: str = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
    AMRBART_MODEL_REVISION: str = "12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512"
    AMRBART_DEVICE: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    AMRBART_MAX_SOURCE_TOKENS: int = 400
    AMRBART_MAX_GRAPH_TOKENS: int = 1024
    AMRBART_NUM_BEAMS: int = 5
    # One text's generation stops here; its output is then dropped, never kept
    # partial (a stopped beam search can return a complete but other graph).
    AMRBART_MAX_SECONDS: float = 45.0
    AMRBART_MAX_INPUT_CHARS: int = 16_384
    AMRBART_BATCH_MAX_INPUTS: int = 64
    AMRBART_RATE_LIMIT_SECONDS: int = 60
    AMRBART_RATE_LIMIT_CALLS: int = 10
    SUMMARY_MODEL_ENABLED: bool = True
    SUMMARY_MODEL_NAME: str = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME: str = "gemma-4-26B_q4_0-it.gguf"
    SUMMARY_MODEL_REVISION: str = "d1c082be9cf3c8a514acf63b8761f4b41935842e"
    SUMMARY_MAX_TOKEN: int = 512
    SUMMARY_TEMP: float = 0.7
    SUMMARY_N_CTX: int = 32768
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    # A JSON request body; uploads are limited by MAX_FILE_SIZE_BYTES instead.
    MAX_REQUEST_BYTES: int = 16 * 1024 * 1024
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
