from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    SUMMARY_ENABLED: bool = True
    SUMMARY_MODEL_NAME: str = "google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME: str = "gemma-3-12b-it-q4_0.gguf"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
