from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    SUMMARY_ENABLED: bool = True
    SUMMARY_MODEL_NAME: str = "google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME: str = "gemma-3-12b-it-q4_0.gguf"
    SUMMARY_MAX_TOKEN: int = 300
    SUMMARY_TEMP: float = 0.2
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
