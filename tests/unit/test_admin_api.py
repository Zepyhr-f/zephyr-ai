import pytest

from app.core.config import Settings
from app.core.diagnostics import diagnostics, provider_status


class FakeDocumentRepository:
    async def count(self):
        return 0

    async def list_documents(self, *, offset=0, limit=1):
        return []


class FakeChunkRepository:
    async def count(self):
        return 0


class FakeSession:
    async def execute(self, statement):
        return None


@pytest.mark.asyncio
async def test_diagnostics_does_not_leak_secrets(monkeypatch) -> None:
    import app.core.diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module, "DocumentRepository", lambda session: FakeDocumentRepository())
    monkeypatch.setattr(diagnostics_module, "ChunkRepository", lambda session: FakeChunkRepository())
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_auth_token="sentinel-token-secret",
        anthropic_base_url="https://user:password@example.test/private/path",
        database_url="postgresql+asyncpg://user:password@db.example/zephyr",
        embedding_provider="ark",
        ark_embedding_api_key="sentinel-embedding-secret",
        ark_embedding_base_url="https://ark.example.test/api/v3",
    )

    response = await diagnostics(FakeSession(), settings)
    dumped = str(response)

    assert response["service"]["name"] == "zephyr-ai"
    assert response["database"] == {"status": "ok"}
    assert "sentinel" not in dumped
    assert "postgresql" not in dumped
    assert "DATABASE_URL" not in dumped
    assert "password" not in dumped


def test_provider_status_uses_whitelisted_fields_only() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sentinel-openai-secret",
        openai_base_url="https://api.example.test/v1",
        embedding_provider="ark",
        ark_embedding_api_key="sentinel-ark-secret",
        ark_embedding_model="embed-model",
    )

    response = provider_status(settings)
    dumped = str(response)

    assert response["llm"]["provider"] == "openai"
    assert response["embedding"]["model"] == "embed-model"
    assert "sentinel" not in dumped
    assert "api.example.test" in dumped
