import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.embedding_client import create_embedding_client
from app.core.llm_client import create_llm_client


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_create_llm_openai_requires_api_key() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        create_llm_client(settings)


def test_create_llm_anthropic_requires_auth_token() -> None:
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key=None,
        anthropic_auth_token=None,
    )

    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN is required"):
        create_llm_client(settings)


def test_create_llm_unsupported_provider() -> None:
    with pytest.raises((ValueError, ValidationError), match="Input should be"):
        Settings(llm_provider="unsupported")


def test_create_embedding_ark_requires_api_key() -> None:
    settings = Settings(embedding_provider="ark", ark_embedding_api_key=None)

    with pytest.raises(ValueError, match="ARK_EMBEDDING_API_KEY is required"):
        create_embedding_client(settings)


def test_create_embedding_unsupported_provider() -> None:
    settings = Settings(embedding_provider="huggingface")

    with pytest.raises(ValueError, match="Unsupported embedding provider: huggingface"):
        create_embedding_client(settings)
