import uuid

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db_session
from app.models import Base, Chunk, Conversation, Document, Message


def test_database_session_factory_uses_async_session() -> None:
    assert AsyncSessionLocal.class_ is AsyncSession


@pytest.mark.asyncio
async def test_get_db_session_yields_async_session() -> None:
    session_generator = get_db_session()
    session = await anext(session_generator)
    try:
        assert isinstance(session, AsyncSession)
    finally:
        await session_generator.aclose()


def test_models_are_registered_with_expected_tables() -> None:
    assert set(Base.metadata.tables) == {
        "chunks",
        "conversations",
        "documents",
        "messages",
    }


def test_chunk_content_vector_uses_pgvector_dimension() -> None:
    vector_type = Chunk.__table__.c.content_vector.type

    assert isinstance(vector_type, Vector)
    assert vector_type.dim == 1536


@pytest.mark.asyncio
async def test_create_document(db_session: AsyncSession) -> None:
    doc = Document(
        file_path="/test/doc.md",
        title="Test Doc",
        metadata_={"author": "test"},
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert isinstance(doc.id, uuid.UUID)
    assert doc.file_path == "/test/doc.md"
    assert doc.title == "Test Doc"
    assert doc.metadata_ == {"author": "test"}
    assert doc.created_at is not None
    assert doc.updated_at is not None


@pytest.mark.asyncio
async def test_document_chunks_relationship(db_session: AsyncSession) -> None:
    doc = Document(file_path="/test/chunked-doc.md", title="Test")
    db_session.add(doc)
    await db_session.flush()

    chunk = Chunk(
        document_id=doc.id,
        content="test content",
        content_vector=[0.1] * 1536,
        chunk_index=0,
        header_path="Root > Section",
        metadata_={"page": 1},
    )
    db_session.add(chunk)
    await db_session.commit()

    result = await db_session.get(Document, doc.id)
    assert result is not None
    await db_session.refresh(result, ["chunks"])

    assert len(result.chunks) == 1
    assert result.chunks[0].content == "test content"
    assert result.chunks[0].document_id == doc.id
    assert result.chunks[0].metadata_ == {"page": 1}


@pytest.mark.asyncio
async def test_conversation_messages_relationship(db_session: AsyncSession) -> None:
    conversation = Conversation(thread_id="thread-1", title="Test Thread")
    db_session.add(conversation)
    await db_session.flush()

    message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="hello",
        tool_calls={"name": "search"},
    )
    db_session.add(message)
    await db_session.commit()

    result = await db_session.get(Conversation, conversation.id)
    assert result is not None
    await db_session.refresh(result, ["messages"])

    assert len(result.messages) == 1
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == "hello"
    assert result.messages[0].tool_calls == {"name": "search"}
    assert result.messages[0].created_at is not None
