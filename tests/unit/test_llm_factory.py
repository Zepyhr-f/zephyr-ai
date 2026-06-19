import pytest
from pydantic import ValidationError

from app.core.config import get_settings


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_get_llm_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_settings_cache()

    from app.core.llm_factory import get_llm

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        get_llm()


def test_get_llm_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clear_settings_cache()

    from app.core.llm_factory import get_llm

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        get_llm()


def test_get_llm_ollama_uses_configured_model_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example:11434")
    clear_settings_cache()

    from app.core.llm_factory import get_llm

    llm = get_llm()

    assert llm.model == "llama3.2"
    assert llm.base_url == "http://ollama.example:11434"


def test_get_llm_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    clear_settings_cache()

    from app.core.llm_factory import get_llm

    with pytest.raises((ValueError, ValidationError), match="Unsupported LLM provider|Input should be"):
        get_llm()


def test_get_embeddings_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_settings_cache()

    from app.core.llm_factory import get_embeddings

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required for embeddings"):
        get_embeddings()


def test_get_embeddings_ollama_uses_configured_model_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example:11434")
    clear_settings_cache()

    from app.core.llm_factory import get_embeddings

    embeddings = get_embeddings()

    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://ollama.example:11434"


def test_get_embeddings_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    clear_settings_cache()

    from app.core.llm_factory import get_embeddings

    with pytest.raises(ValueError, match="Unsupported embedding provider: huggingface"):
        get_embeddings()
