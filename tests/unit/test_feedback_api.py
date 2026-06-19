from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.feedback.schemas import FeedbackCreateRequest
from app.feedback.service import FeedbackService


class FakeScalarResult:
    def __init__(self, scalar=None, rows=None):
        self.scalar = scalar
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.scalar

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, conversation, message=None):
        self.conversation = conversation
        self.message = message
        self.added = []

    async def execute(self, statement):
        return FakeScalarResult(self.conversation)

    async def scalar(self, statement):
        return len(self.added)

    async def get(self, model, key):
        return self.message

    def add(self, instance):
        instance.id = uuid4()
        instance.created_at = datetime.now(timezone.utc)
        self.added.append(instance)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_feedback_create_validates_message_belongs_to_conversation() -> None:
    conversation = SimpleNamespace(id=uuid4(), thread_id="thread-1")
    message = SimpleNamespace(id=uuid4(), conversation_id=uuid4())
    service = FeedbackService(session=FakeSession(conversation, message))

    with pytest.raises(HTTPException) as exc:
        await service.create(
            FeedbackCreateRequest(
                conversation_id="thread-1",
                message_id=str(message.id),
                rating="up",
            )
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_feedback_create_returns_public_thread_id() -> None:
    conversation = SimpleNamespace(id=uuid4(), thread_id="thread-1")
    message = SimpleNamespace(id=uuid4(), conversation_id=conversation.id)
    service = FeedbackService(session=FakeSession(conversation, message))

    response = await service.create(
        FeedbackCreateRequest(
            conversation_id="thread-1",
            message_id=str(message.id),
            rating="down",
            comment="不准确",
            source="web",
            metadata={"reason": "stale"},
        )
    )

    assert response.conversation_id == "thread-1"
    assert response.message_id == str(message.id)
    assert response.rating == "down"
    assert response.metadata == {"reason": "stale"}


@pytest.mark.asyncio
async def test_feedback_missing_conversation_returns_404() -> None:
    service = FeedbackService(session=FakeSession(None))

    with pytest.raises(HTTPException) as exc:
        await service.create(FeedbackCreateRequest(conversation_id="missing", rating="neutral"))

    assert exc.value.status_code == 404
