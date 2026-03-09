from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "tiny-llama-chat-inference"
    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    redis_url: str = "redis://redis:6379/0"
    redis_history_ttl_seconds: int = Field(default=60 * 60 * 24)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
