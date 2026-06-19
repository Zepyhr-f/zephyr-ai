from datetime import datetime

from pydantic import BaseModel, Field

from app.models.feedback import FeedbackRating


class FeedbackCreateRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    message_id: str | None = None
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=4000)
    source: str | None = Field(default=None, max_length=100)
    metadata: dict | None = None


class FeedbackResponse(BaseModel):
    id: str
    conversation_id: str
    message_id: str | None
    rating: FeedbackRating
    comment: str | None
    source: str | None
    metadata: dict | None
    created_at: datetime


class FeedbackRatingCount(BaseModel):
    rating: FeedbackRating
    count: int


class FeedbackSourceCount(BaseModel):
    source: str | None
    count: int


class FeedbackStats(BaseModel):
    total: int
    ratings: list[FeedbackRatingCount]
    sources: list[FeedbackSourceCount]
