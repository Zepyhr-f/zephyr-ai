from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_auth_token: str | None = None
    anthropic_base_url: str = "https://ark.cn-beijing.volces.com/api/coding"
    anthropic_model: str = "ark-code-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Embedding
    embedding_provider: Literal["openai", "huggingface", "ollama", "ark"] = "ark"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    ark_embedding_api_key: str | None = None
    ark_embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_embedding_model: str = "doubao-embedding"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/zephyr_ai"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def is_openai(self) -> bool:
        return self.llm_provider == "openai"

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider == "anthropic"

    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == "ollama"


@lru_cache
def get_settings() -> Settings:
    return Settings()
