from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WORKBENCH_API_KEY: str | None = None
    SUMMARY_ENABLED: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
