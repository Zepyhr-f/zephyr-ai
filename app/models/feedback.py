import uuid
from enum import StrEnum

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class Feedback(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "feedback"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        SAEnum(FeedbackRating, name="feedback_rating", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(100))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)

    conversation = relationship("Conversation", lazy="selectin")
    message = relationship("Message", lazy="selectin")
