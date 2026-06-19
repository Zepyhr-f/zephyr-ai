from datetime import datetime

from pydantic import BaseModel, Field

from app.rag.schemas import RagSource


class KnowledgeStats(BaseModel):
    document_count: int
    chunk_count: int
    last_ingested_at: datetime | None


class KnowledgeDocumentSummary(BaseModel):
    id: str
    file_path: str
    title: str | None
    metadata: dict | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class KnowledgeDocumentList(BaseModel):
    items: list[KnowledgeDocumentSummary]
    total: int
    offset: int
    limit: int


class KnowledgeDocumentDetail(KnowledgeDocumentSummary):
    pass


class KnowledgeChunkDetail(BaseModel):
    id: str
    document_id: str
    document_title: str | None
    document_file_path: str
    content: str
    header_path: str | None
    metadata: dict | None
    chunk_index: int
    preview: str
    created_at: datetime
    updated_at: datetime


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResponse(BaseModel):
    query: str
    top_k: int
    sources: list[RagSource]
