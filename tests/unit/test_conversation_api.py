from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.conversations.service import ConversationService, _safe_sources


class FakeScalarResult:
    def __init__(self, *, scalar=None, scalars=None):
        self.scalar = scalar
        self.scalars_value = scalars or []

    def scalar_one_or_none(self):
        return self.scalar

    def scalars(self):
        return SimpleNamespace(all=lambda: self.scalars_value)


class FakeSession:
    def __init__(self, conversation, messages):
        self.conversation = conversation
        self.messages = messages
        self.execute_calls = 0

    async def execute(self, statement):
        self.execute_calls += 1
        if self.execute_calls in {1, 2}:
            return FakeScalarResult(scalar=self.conversation)
        return FakeScalarResult(scalars=self.messages)

    async def scalar(self, statement):
        return len(self.messages)


def test_safe_sources_returns_empty_for_bad_payloads() -> None:
    assert _safe_sources(None) == []
    assert _safe_sources({"sources": "bad"}) == []
    assert _safe_sources({"sources": [{"bad": "shape"}]}) == []


def test_safe_sources_parses_valid_sources() -> None:
    sources = _safe_sources(
        {
            "sources": [
                {
                    "title": "架构",
                    "file_path": "docs/arch.md",
                    "header_path": "数据库",
                    "score": 0.9,
                    "preview": "pgvector",
                    "chunk_index": 1,
                }
            ]
        }
    )

    assert sources[0].title == "架构"


@pytest.mark.asyncio
async def test_conversation_service_uses_thread_id_and_lists_messages() -> None:
    now = datetime.now(timezone.utc)
    conversation = SimpleNamespace(
        id=uuid4(),
        thread_id="thread-1",
        title="问题",
        created_at=now,
        updated_at=now,
    )
    messages = [
        SimpleNamespace(id=uuid4(), role="user", content="问", tool_calls=None, created_at=now),
        SimpleNamespace(
            id=uuid4(),
            role="assistant",
            content="答",
            tool_calls={"sources": []},
            created_at=now,
        ),
    ]
    service = ConversationService(session=FakeSession(conversation, messages))

    detail = await service.get_conversation("thread-1")
    listing = await service.list_messages(conversation_id="thread-1")

    assert detail.id == "thread-1"
    assert detail.message_count == 2
    assert [message.role for message in listing.items] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_missing_conversation_returns_404() -> None:
    service = ConversationService(session=FakeSession(None, []))

    with pytest.raises(HTTPException) as exc:
        await service.get_conversation("missing")

    assert exc.value.status_code == 404
