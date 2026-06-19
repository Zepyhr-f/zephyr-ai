from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.embedding_client import create_embedding_client
from app.knowledge.schemas import (
    KnowledgeChunkDetail,
    KnowledgeDocumentDetail,
    KnowledgeDocumentList,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeStats,
)
from app.knowledge.service import KnowledgeService
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.service import SemanticRetrievalService

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def _knowledge_service(session: AsyncSession, *, with_retrieval: bool = False) -> KnowledgeService:
    chunk_repository = ChunkRepository(session)
    retrieval_service = None
    if with_retrieval:
        settings = get_settings()
        retrieval_service = SemanticRetrievalService(
            embedding_client=create_embedding_client(settings),
            repository=chunk_repository,
        )
    return KnowledgeService(
        document_repository=DocumentRepository(session),
        chunk_repository=chunk_repository,
        retrieval_service=retrieval_service,
    )


@router.get("/stats", response_model=KnowledgeStats)
async def stats(session: AsyncSession = Depends(get_db_session)) -> KnowledgeStats:
    return await _knowledge_service(session).stats()


@router.get("/documents", response_model=KnowledgeDocumentList)
async def documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeDocumentList:
    return await _knowledge_service(session).list_documents(offset=offset, limit=limit)


@router.get("/documents/{document_id}", response_model=KnowledgeDocumentDetail)
async def document(
    document_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeDocumentDetail:
    return await _knowledge_service(session).get_document(document_id)


@router.get("/chunks/{chunk_id}", response_model=KnowledgeChunkDetail)
async def chunk(
    chunk_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeChunkDetail:
    return await _knowledge_service(session).get_chunk(chunk_id)


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search(
    request: KnowledgeSearchRequest,
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeSearchResponse:
    return await _knowledge_service(session, with_retrieval=True).search(
        query=request.query,
        top_k=request.top_k,
    )
