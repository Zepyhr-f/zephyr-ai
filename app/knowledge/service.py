import uuid

from fastapi import HTTPException

from app.knowledge.schemas import (
    KnowledgeChunkDetail,
    KnowledgeDocumentDetail,
    KnowledgeDocumentList,
    KnowledgeDocumentSummary,
    KnowledgeSearchResponse,
    KnowledgeStats,
)
from app.models import Chunk, Document
from app.rag.prompt import source_from_result
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.service import SemanticRetrievalService


class KnowledgeService:
    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        retrieval_service: SemanticRetrievalService | None = None,
    ) -> None:
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.retrieval_service = retrieval_service

    async def stats(self) -> KnowledgeStats:
        documents = await self.document_repository.list_documents(offset=0, limit=1)
        return KnowledgeStats(
            document_count=await self.document_repository.count(),
            chunk_count=await self.chunk_repository.count(),
            last_ingested_at=documents[0].updated_at if documents else None,
        )

    async def list_documents(self, *, offset: int = 0, limit: int = 50) -> KnowledgeDocumentList:
        documents = await self.document_repository.list_documents(offset=offset, limit=limit)
        return KnowledgeDocumentList(
            items=[await self._document_summary(document) for document in documents],
            total=await self.document_repository.count(),
            offset=offset,
            limit=limit,
        )

    async def get_document(self, document_id: str) -> KnowledgeDocumentDetail:
        document = await self.document_repository.get(_parse_uuid(document_id, "document_id"))
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        summary = await self._document_summary(document)
        return KnowledgeDocumentDetail(**summary.model_dump())

    async def get_chunk(self, chunk_id: str) -> KnowledgeChunkDetail:
        chunk = await self.chunk_repository.get(_parse_uuid(chunk_id, "chunk_id"))
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return self._chunk_detail(chunk)

    async def search(self, *, query: str, top_k: int = 5) -> KnowledgeSearchResponse:
        if self.retrieval_service is None:
            raise RuntimeError("retrieval_service is required for search")
        normalized_query = query.strip()
        results = await self.retrieval_service.search(normalized_query, limit=top_k)
        return KnowledgeSearchResponse(
            query=normalized_query,
            top_k=top_k,
            sources=[source_from_result(result) for result in results],
        )

    async def _document_summary(self, document: Document) -> KnowledgeDocumentSummary:
        return KnowledgeDocumentSummary(
            id=str(document.id),
            file_path=document.file_path,
            title=document.title,
            metadata=document.metadata_,
            chunk_count=await self.chunk_repository.count_by_document(document.id),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def _chunk_detail(self, chunk: Chunk) -> KnowledgeChunkDetail:
        document = chunk.document
        return KnowledgeChunkDetail(
            id=str(chunk.id),
            document_id=str(chunk.document_id),
            document_title=document.title if document else None,
            document_file_path=document.file_path if document else "",
            content=chunk.content,
            header_path=chunk.header_path,
            metadata=chunk.metadata_,
            chunk_index=chunk.chunk_index,
            preview=chunk.content.replace("\n", " ")[:200],
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
        )


def _parse_uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}") from exc
