from pydantic import BaseModel, Field


class RagChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0, le=1)
    max_tokens: int = Field(default=1024, ge=128, le=4096)
    include_history: bool = True


class RagSource(BaseModel):
    title: str | None
    file_path: str
    header_path: str | None
    score: float
    preview: str
    chunk_index: int | None = None


class RagChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[RagSource]
