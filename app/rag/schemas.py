from pydantic import BaseModel, Field


class RagChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class RagSource(BaseModel):
    title: str | None
    file_path: str
    header_path: str | None
    score: float
    preview: str


class RagChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[RagSource]
