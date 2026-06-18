import logging

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging


def test_settings_defaults():
    s = Settings()
    assert s.llm_provider == "openai"
    assert s.openai_model == "gpt-4o-mini"
    assert s.embedding_dimension == 1536


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")

    s = Settings()
    assert s.llm_provider == "anthropic"
    assert s.anthropic_model == "claude-opus-4-7"


def test_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_setup_logging_configures_root_logger(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    setup_logging()

    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_preserves_existing_handlers(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    get_settings.cache_clear()
    existing_handler = logging.NullHandler()
    monkeypatch.setattr(logging.getLogger(), "handlers", [existing_handler])

    setup_logging()

    assert existing_handler in logging.getLogger().handlers
