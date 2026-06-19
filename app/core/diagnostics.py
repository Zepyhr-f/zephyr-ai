from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.version import SERVICE_NAME, SERVICE_VERSION
from app.knowledge.service import KnowledgeService
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository


def _host(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.hostname:
        return parsed.hostname
    return parsed.path or None


def provider_status(settings: Settings) -> dict:
    return {
        "llm": {
            "provider": settings.llm_provider,
            "model": _llm_model(settings),
            "base_url_host": _host(_llm_base_url(settings)),
            "configured": _llm_configured(settings),
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": _embedding_model(settings),
            "base_url_host": _host(_embedding_base_url(settings)),
            "embedding_dim": settings.embedding_dimension,
            "configured": _embedding_configured(settings),
        },
    }


async def diagnostics(session: AsyncSession, settings: Settings) -> dict:
    database = await _database_status(session)
    knowledge = await KnowledgeService(
        document_repository=DocumentRepository(session),
        chunk_repository=ChunkRepository(session),
    ).stats()
    return {
        "service": {"name": SERVICE_NAME, "version": SERVICE_VERSION},
        "database": database,
        "knowledge": knowledge.model_dump(mode="json"),
        "providers": provider_status(settings),
    }


async def _database_status(session: AsyncSession) -> dict[str, str]:
    try:
        await session.execute(text("select 1"))
        return {"status": "ok"}
    except Exception:
        return {"status": "unavailable"}


def _llm_model(settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return settings.openai_model
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    return settings.anthropic_model


def _llm_base_url(settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return settings.openai_base_url
    if settings.llm_provider == "ollama":
        return settings.ollama_base_url
    return settings.anthropic_base_url


def _llm_configured(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    if settings.llm_provider == "ollama":
        return bool(settings.ollama_base_url)
    return bool(settings.anthropic_auth_token or settings.anthropic_api_key)


def _embedding_model(settings: Settings) -> str:
    if settings.embedding_provider == "ark":
        return settings.ark_embedding_model
    return settings.embedding_model


def _embedding_base_url(settings: Settings) -> str | None:
    if settings.embedding_provider == "ark":
        return settings.ark_embedding_base_url
    if settings.embedding_provider == "ollama":
        return settings.ollama_base_url
    return None


def _embedding_configured(settings: Settings) -> bool:
    if settings.embedding_provider == "ark":
        return bool(settings.ark_embedding_api_key)
    if settings.embedding_provider == "ollama":
        return bool(settings.ollama_base_url)
    return settings.embedding_provider in {"huggingface", "openai"}
