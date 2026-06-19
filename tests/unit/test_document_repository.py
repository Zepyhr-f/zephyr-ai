import uuid

import pytest
from sqlalchemy.sql.dml import Delete
from sqlalchemy.sql.selectable import Select

from app.ingestion.markdown import MarkdownChunk, MarkdownDocument
from app.ingestion.service import StoredDocumentChunk
from app.models import Chunk, Document
from app.repositories.document_repository import DocumentRepository


class FakeAsyncSession:
    def __init__(self, existing_document: Document | None = None) -> None:
        self.existing_document = existing_document
        self.added: list[object] = []
        self.executed: list[object] = []
        self.flushed = 0
        self.committed = False

    async def scalar(self, statement: object) -> Document | None:
        self.executed.append(statement)
        return self.existing_document

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object) -> None:
        self.executed.append(statement)

    async def flush(self) -> None:
        self.flushed += 1

    async def commit(self) -> None:
        self.committed = True
        raise AssertionError("repository must not commit")


def make_document() -> MarkdownDocument:
    return MarkdownDocument(
        file_path="docs/test.md",
        title="Test",
        content="# Test\nBody",
        metadata={"source_path": "docs/test.md", "category": "docs"},
    )


def make_chunk_vectors() -> list[tuple[MarkdownChunk, list[float]]]:
    return [
        (
            MarkdownChunk(
                chunk_index=0,
                content="first",
                header_path="Intro",
                metadata={"chunk_index": 0, "header_path": "Intro"},
            ),
            [0.1, 0.2],
        ),
        (
            MarkdownChunk(
                chunk_index=1,
                content="second",
                header_path=None,
                metadata={"chunk_index": 1, "header_path": None},
            ),
            [0.3, 0.4],
        ),
    ]


@pytest.mark.asyncio
async def test_upsert_document_with_chunks_creates_document_and_chunks() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)

    stored_chunks = await repository.upsert_document_with_chunks(make_document(), make_chunk_vectors())

    document = session.added[0]
    assert isinstance(document, Document)
    assert document.file_path == "docs/test.md"
    assert document.title == "Test"
    assert document.metadata_ == {"source_path": "docs/test.md", "category": "docs"}

    chunks = session.added[1:]
    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].document == document
    assert chunks[0].content == "first"
    assert chunks[0].content_vector == [0.1, 0.2]
    assert chunks[0].chunk_index == 0
    assert chunks[0].header_path == "Intro"
    assert chunks[0].metadata_ == {"chunk_index": 0, "header_path": "Intro"}
    assert chunks[1].document == document
    assert chunks[1].content == "second"
    assert chunks[1].content_vector == [0.3, 0.4]
    assert chunks[1].chunk_index == 1
    assert chunks[1].header_path is None
    assert chunks[1].metadata_ == {"chunk_index": 1, "header_path": None}
    assert stored_chunks == [
        StoredDocumentChunk(chunk=make_chunk_vectors()[0][0], vector=[0.1, 0.2]),
        StoredDocumentChunk(chunk=make_chunk_vectors()[1][0], vector=[0.3, 0.4]),
    ]
    assert session.flushed >= 1
    assert session.committed is False


@pytest.mark.asyncio
async def test_upsert_document_with_chunks_updates_existing_document_and_deletes_old_chunks() -> None:
    existing = Document(
        id=uuid.uuid4(),
        file_path="docs/test.md",
        title="Old",
        metadata_={"old": True},
    )
    session = FakeAsyncSession(existing_document=existing)
    repository = DocumentRepository(session)

    await repository.upsert_document_with_chunks(make_document(), make_chunk_vectors())

    assert existing.title == "Test"
    assert existing.metadata_ == {"source_path": "docs/test.md", "category": "docs"}
    assert any(isinstance(statement, Delete) for statement in session.executed)
    inserted_chunks = [item for item in session.added if isinstance(item, Chunk)]
    assert len(inserted_chunks) == 2
    assert all(chunk.document == existing for chunk in inserted_chunks)
    assert session.committed is False


@pytest.mark.asyncio
async def test_upsert_document_with_chunks_queries_by_file_path() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)

    await repository.upsert_document_with_chunks(make_document(), [])

    select_statement = session.executed[0]
    assert isinstance(select_statement, Select)
    assert "documents.file_path" in str(select_statement)
    assert session.committed is False
