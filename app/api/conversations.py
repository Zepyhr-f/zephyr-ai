from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.schemas import ConversationDetail, ConversationMessageList
from app.conversations.service import ConversationService
from app.core.database import get_db_session

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ConversationDetail:
    return await ConversationService(session=session).get_conversation(conversation_id)


@router.get("/{conversation_id}/messages", response_model=ConversationMessageList)
async def messages(
    conversation_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationMessageList:
    return await ConversationService(session=session).list_messages(
        conversation_id=conversation_id,
        offset=offset,
        limit=limit,
    )
