import uuid

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.feedback.schemas import (
    FeedbackCreateRequest,
    FeedbackRatingCount,
    FeedbackResponse,
    FeedbackSourceCount,
    FeedbackStats,
)
from app.models import Conversation, Feedback, Message


class FeedbackService:
    def __init__(self, *, session: AsyncSession) -> None:
        self.session = session

    async def create(self, request: FeedbackCreateRequest) -> FeedbackResponse:
        conversation = await self._conversation(request.conversation_id)
        message = None
        if request.message_id:
            message = await self._message(request.message_id)
            if message.conversation_id != conversation.id:
                raise HTTPException(status_code=400, detail="Message does not belong to conversation")

        feedback = Feedback(
            conversation_id=conversation.id,
            message_id=message.id if message else None,
            rating=request.rating,
            comment=request.comment,
            source=request.source,
            metadata_=request.metadata,
        )
        self.session.add(feedback)
        await self.session.flush()
        return self._response(feedback, conversation.thread_id)

    async def stats(self) -> FeedbackStats:
        total = int(await self.session.scalar(select(func.count()).select_from(Feedback)) or 0)
        rating_rows = (
            await self.session.execute(
                select(Feedback.rating, func.count()).group_by(Feedback.rating).order_by(Feedback.rating)
            )
        ).all()
        source_rows = (
            await self.session.execute(
                select(Feedback.source, func.count()).group_by(Feedback.source).order_by(Feedback.source)
            )
        ).all()
        return FeedbackStats(
            total=total,
            ratings=[FeedbackRatingCount(rating=row[0], count=int(row[1])) for row in rating_rows],
            sources=[FeedbackSourceCount(source=row[0], count=int(row[1])) for row in source_rows],
        )

    async def _conversation(self, conversation_id: str) -> Conversation:
        result = await self.session.execute(
            select(Conversation).where(Conversation.thread_id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    async def _message(self, message_id: str) -> Message:
        try:
            message_uuid = uuid.UUID(message_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid message_id") from exc
        message = await self.session.get(Message, message_uuid)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found")
        return message

    def _response(self, feedback: Feedback, conversation_id: str) -> FeedbackResponse:
        return FeedbackResponse(
            id=str(feedback.id),
            conversation_id=conversation_id,
            message_id=str(feedback.message_id) if feedback.message_id else None,
            rating=feedback.rating,
            comment=feedback.comment,
            source=feedback.source,
            metadata=feedback.metadata_,
            created_at=feedback.created_at,
        )
