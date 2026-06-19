from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.knowledge.service import KnowledgeService
from app.retrieval.service import SemanticSearchResult


class FakeDocumentRepository:
    def __init__(self, documents):
        self.documents = documents

    async def count(self):
        return len(self.documents)

    async def list_documents(self, *, offset=0, limit=50):
        return self.documents[offset : offset + limit]

    async def get(self, document_id):
        return next((document for document in self.documents if document.id == document_id), None)


class FakeChunkRepository:
    def __init__(self, chunks):
        self.chunks = chunks

    async def count(self):
        return len(self.chunks)

    async def count_by_document(self, document_id):
        return len([chunk for chunk in self.chunks if chunk.document_id == document_id])

    async def get(self, chunk_id):
        return next((chunk for chunk in self.chunks if chunk.id == chunk_id), None)


class FakeRetrievalService:
    async def search(self, query, limit=5):
        return [
            SemanticSearchResult(
                content="Zephyr AI 使用 pgvector。",
                header_path="数据库",
                document_title="架构",
                document_file_path="docs/arch.md",
                metadata={"chunk_index": 2},
                distance=0.1,
                score=0.9,
                chunk_id="chunk-1",
                document_id="doc-1",
            )
        ]


def make_document():
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        file_path="docs/arch.md",
        title="架构",
        metadata_={"scope": "ai"},
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_knowledge_stats_and_document_listing() -> None:
    document = make_document()
    chunks = [SimpleNamespace(id=uuid4(), document_id=document.id)]
    service = KnowledgeService(
        document_repository=FakeDocumentRepository([document]),
        chunk_repository=FakeChunkRepository(chunks),
    )

    stats = await service.stats()
    listing = await service.list_documents()

    assert stats.document_count == 1
    assert stats.chunk_count == 1
    assert stats.last_ingested_at == document.updated_at
    assert listing.items[0].file_path == "docs/arch.md"
    assert listing.items[0].chunk_count == 1


@pytest.mark.asyncio
async def test_knowledge_chunk_detail_does_not_include_vector() -> None:
    document = make_document()
    chunk = SimpleNamespace(
        id=uuid4(),
        document_id=document.id,
        document=document,
        content="Zephyr AI 使用 PostgreSQL / pgvector 存储文档向量。",
        header_path="数据库",
        metadata_={"chunk_index": 1},
        chunk_index=1,
        created_at=document.created_at,
        updated_at=document.updated_at,
        content_vector=[0.1, 0.2],
    )
    service = KnowledgeService(
        document_repository=FakeDocumentRepository([document]),
        chunk_repository=FakeChunkRepository([chunk]),
    )

    response = await service.get_chunk(str(chunk.id))

    dumped = response.model_dump()
    assert dumped["preview"].startswith("Zephyr AI")
    assert "content_vector" not in dumped


@pytest.mark.asyncio
async def test_knowledge_search_reuses_semantic_retrieval() -> None:
    service = KnowledgeService(
        document_repository=FakeDocumentRepository([]),
        chunk_repository=FakeChunkRepository([]),
        retrieval_service=FakeRetrievalService(),
    )

    response = await service.search(query=" pgvector ", top_k=3)

    assert response.query == "pgvector"
    assert response.top_k == 3
    assert response.sources[0].score == 0.9
