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
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_auth_token: str | None = None
    anthropic_base_url: str = "https://ark.cn-beijing.volces.com/api/coding"
    anthropic_model: str = "ark-code-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Embedding
    embedding_provider: Literal["openai", "huggingface", "ollama", "ark"] = "ark"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 2048
    ark_embedding_api_key: str | None = None
    ark_embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_embedding_endpoint: str = "/embeddings/multimodal"
    ark_embedding_model: str = "ep-20260619185149-f9c8c"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/zephyr_ai"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Logging — file output. dir 不存在或无权限时自动 fallback 为仅 stdout。
    log_dir: str = "/app/logs/ai"
    log_file_enable: bool = True
    log_file_max_bytes: int = 50 * 1024 * 1024  # 50MB
    log_file_backup_count: int = 14

    # Gateway sign (replay protection + HMAC) — enforced for every request
    # except the configured health-exempt paths.
    zephyr_gateway_sign_secret: str | None = None
    zephyr_gateway_sign_ttl_seconds: int = 300
    zephyr_gateway_redis_host: str = "redis:6379"
    zephyr_gateway_redis_password: str | None = None
    zephyr_gateway_redis_db: int = 0
    zephyr_gateway_nonce_prefix: str = "gw:nonce:"
    zephyr_gateway_health_exempt: str = "/health"
    zephyr_gateway_sign_required: bool = True

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
