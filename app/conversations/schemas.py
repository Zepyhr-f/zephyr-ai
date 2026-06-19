from datetime import datetime

from pydantic import BaseModel

from app.rag.schemas import RagSource


class ConversationDetail(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: list[RagSource]
    created_at: datetime


class ConversationMessageList(BaseModel):
    items: list[ConversationMessage]
    total: int
    offset: int
    limit: int
