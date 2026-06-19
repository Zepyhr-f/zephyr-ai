from types import SimpleNamespace

import pytest

from app.retrieval.service import SemanticRetrievalService, SemanticSearchResult


class FakeEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[str] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector


class FakeRepository:
    def __init__(self, results: list[SemanticSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[float], int]] = []

    async def search_similar(
        self,
        vector: list[float],
        limit: int,
    ) -> list[SemanticSearchResult]:
        self.calls.append((vector, limit))
        return self.results


@pytest.mark.asyncio
async def test_semantic_retrieval_trims_query_and_normalizes_vector() -> None:
    result = SemanticSearchResult(
        content="chunk content",
        header_path="Intro > Details",
        document_title="Document title",
        document_file_path="docs/example.md",
        metadata={"chunk": 1},
        distance=0.25,
        score=0.75,
    )
    embedding_client = FakeEmbeddingClient([3.0, 4.0])
    repository = FakeRepository([result])
    service = SemanticRetrievalService(
        embedding_client=embedding_client,
        repository=repository,
    )

    results = await service.search("  alpha beta  ", limit=3)

    assert embedding_client.queries == ["alpha beta"]
    assert repository.calls == [([0.6, 0.8], 3)]
    assert results == [result]


@pytest.mark.asyncio
async def test_semantic_retrieval_rejects_empty_query() -> None:
    service = SemanticRetrievalService(
        embedding_client=FakeEmbeddingClient([1.0]),
        repository=FakeRepository(),
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        await service.search("  ")


@pytest.mark.asyncio
async def test_semantic_retrieval_rejects_invalid_limit() -> None:
    service = SemanticRetrievalService(
        embedding_client=FakeEmbeddingClient([1.0]),
        repository=FakeRepository(),
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await service.search("alpha", limit=0)


@pytest.mark.asyncio
async def test_chunk_repository_search_similar_returns_result_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.repositories.chunk_repository import ChunkRepository

    captured_statement = None

    class FakeExecuteResult:
        def all(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    id="chunk-1",
                    document_id="document-1",
                    content="chunk content",
                    header_path="Intro",
                    metadata_={"source": "unit"},
                    title="Doc title",
                    file_path="docs/doc.md",
                    distance=0.2,
                )
            ]

    class FakeSession:
        async def execute(self, statement):
            nonlocal captured_statement
            captured_statement = statement
            return FakeExecuteResult()

    repository = ChunkRepository(FakeSession())
    results = await repository.search_similar([0.6, 0.8], 2)

    assert captured_statement is not None
    statement_text = str(captured_statement)
    assert "ORDER BY distance" in statement_text
    assert "LIMIT" in statement_text
    assert results == [
        SemanticSearchResult(
            content="chunk content",
            header_path="Intro",
            document_title="Doc title",
            document_file_path="docs/doc.md",
            metadata={"source": "unit"},
            distance=0.2,
            score=0.8,
            chunk_id="chunk-1",
            document_id="document-1",
        )
    ]
