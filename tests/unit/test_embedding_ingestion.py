from pathlib import Path

import pytest

from app.ingestion.markdown import MarkdownChunk, MarkdownDocument
from app.ingestion.service import EmbeddingIngestionError, EmbeddingIngestionService


class FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.vectors is not None:
            return self.vectors
        return [[float(index + 1)] for index, _ in enumerate(texts)]


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def upsert_document_with_chunks(
        self,
        document: MarkdownDocument,
        chunk_vectors: list,
    ) -> list:
        self.calls.append({"document": document, "chunk_vectors": chunk_vectors})
        return [
            {"chunk": chunk, "vector": vector}
            for chunk, vector in chunk_vectors
        ]


def make_document() -> MarkdownDocument:
    return MarkdownDocument(
        file_path="docs/test.md",
        title="Test",
        content="# Test\nBody",
        metadata={"source_path": "docs/test.md"},
    )


def make_chunks() -> list[MarkdownChunk]:
    return [
        MarkdownChunk(
            chunk_index=0,
            content="first",
            header_path="Intro",
            metadata={"chunk_index": 0, "header_path": "Intro"},
        ),
        MarkdownChunk(
            chunk_index=1,
            content="second",
            header_path="Usage",
            metadata={"chunk_index": 1, "header_path": "Usage"},
        ),
    ]


@pytest.mark.asyncio
async def test_ingest_document_embeds_texts_in_chunk_order() -> None:
    embedding_client = FakeEmbeddingClient([[0.1], [0.2]])
    store = FakeStore()
    service = EmbeddingIngestionService(embedding_client=embedding_client, store=store)

    await service.ingest_document(make_document(), make_chunks())

    assert embedding_client.calls == [["first", "second"]]


@pytest.mark.asyncio
async def test_ingest_document_rejects_vector_count_mismatch_without_store_call() -> None:
    embedding_client = FakeEmbeddingClient([[0.1]])
    store = FakeStore()
    service = EmbeddingIngestionService(embedding_client=embedding_client, store=store)

    with pytest.raises(EmbeddingIngestionError, match="vector count"):
        await service.ingest_document(make_document(), make_chunks())

    assert store.calls == []


@pytest.mark.asyncio
async def test_ingest_document_passes_chunk_fields_and_vectors_to_store() -> None:
    document = make_document()
    chunks = make_chunks()
    vectors = [[3.0, 4.0], [0.0, 2.0]]
    store = FakeStore()
    service = EmbeddingIngestionService(
        embedding_client=FakeEmbeddingClient(vectors),
        store=store,
    )

    result = await service.ingest_document(document, chunks)

    assert result.document == document
    assert result.chunk_count == 2
    assert store.calls == [
        {
            "document": document,
            "chunk_vectors": [
                (chunks[0], [0.6, 0.8]),
                (chunks[1], [0.0, 1.0]),
            ],
        }
    ]
    assert store.calls[0]["chunk_vectors"][0][0].chunk_index == 0
    assert store.calls[0]["chunk_vectors"][0][0].content == "first"
    assert store.calls[0]["chunk_vectors"][0][0].header_path == "Intro"
    assert store.calls[0]["chunk_vectors"][0][0].metadata == {"chunk_index": 0, "header_path": "Intro"}
    assert result.stored_chunks == [
        {"chunk": chunks[0], "vector": [0.6, 0.8]},
        {"chunk": chunks[1], "vector": [0.0, 1.0]},
    ]


@pytest.mark.asyncio
async def test_ingest_document_rejects_zero_embedding_vector_without_store_call() -> None:
    embedding_client = FakeEmbeddingClient([[0.0, 0.0], [1.0, 0.0]])
    store = FakeStore()
    service = EmbeddingIngestionService(embedding_client=embedding_client, store=store)

    with pytest.raises(EmbeddingIngestionError, match="normalization failed"):
        await service.ingest_document(make_document(), make_chunks())

    assert store.calls == []


@pytest.mark.asyncio
async def test_ingest_file_loads_and_splits_tmp_markdown(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# One\nAlpha\n\n# Two\nBeta", encoding="utf-8")
    service = EmbeddingIngestionService(
        embedding_client=FakeEmbeddingClient([[0.1], [0.2]]),
        store=FakeStore(),
    )

    result = await service.ingest_file(path, chunk_size=100, chunk_overlap=0)

    assert result.document.file_path == str(path)
    assert result.chunk_count == 2


@pytest.mark.asyncio
async def test_ingest_directory_uses_root_and_summarizes_counts(tmp_path: Path) -> None:
    first = tmp_path / "b.md"
    second = tmp_path / "a.md"
    first.write_text("# B\nBeta", encoding="utf-8")
    second.write_text("# A\nAlpha\n\n# More\nGamma", encoding="utf-8")
    service = EmbeddingIngestionService(
        embedding_client=FakeEmbeddingClient(),
        store=FakeStore(),
    )

    result = await service.ingest_directory(root=tmp_path, chunk_size=100, chunk_overlap=0)

    assert result.document_count == 2
    assert result.chunk_count == 3
    assert [item.document.file_path for item in result.results] == [str(second), str(first)]
