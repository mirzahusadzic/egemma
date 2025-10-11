from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    SUMMARY_ENABLED: bool = True
    SUMMARY_MODEL_NAME: str = "google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME: str = "gemma-3-12b-it-q4_0.gguf"
    SUMMARY_MAX_TOKEN: int = 300
    SUMMARY_TEMP: float = 0.2
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
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
