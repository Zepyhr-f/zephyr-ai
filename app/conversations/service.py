from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.schemas import ConversationDetail, ConversationMessage, ConversationMessageList
from app.models import Conversation, Message
from app.rag.schemas import RagSource


class ConversationService:
    def __init__(self, *, session: AsyncSession) -> None:
        self.session = session

    async def get_conversation(self, conversation_id: str) -> ConversationDetail:
        conversation = await self._conversation(conversation_id)
        return ConversationDetail(
            id=conversation.thread_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=await self._message_count(conversation.id),
        )

    async def list_messages(
        self,
        *,
        conversation_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> ConversationMessageList:
        conversation = await self._conversation(conversation_id)
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        messages = list(result.scalars().all())
        return ConversationMessageList(
            items=[self._message_schema(message) for message in messages],
            total=await self._message_count(conversation.id),
            offset=offset,
            limit=limit,
        )

    async def _conversation(self, conversation_id: str) -> Conversation:
        result = await self.session.execute(
            select(Conversation).where(Conversation.thread_id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    async def _message_count(self, conversation_db_id) -> int:
        return int(
            await self.session.scalar(
                select(func.count()).select_from(Message).where(Message.conversation_id == conversation_db_id)
            )
            or 0
        )

    def _message_schema(self, message: Message) -> ConversationMessage:
        return ConversationMessage(
            id=str(message.id),
            role=message.role,
            content=message.content,
            sources=_safe_sources(message.tool_calls),
            created_at=message.created_at,
        )


def _safe_sources(tool_calls: dict | None) -> list[RagSource]:
    if not isinstance(tool_calls, dict):
        return []
    raw_sources = tool_calls.get("sources")
    if not isinstance(raw_sources, list):
        return []
    sources: list[RagSource] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            continue
        try:
            sources.append(RagSource(**raw_source))
        except ValueError:
            continue
    return sources
