from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.embedding_client import create_embedding_client
from app.core.llm_client import LLMProviderError, create_llm_client
from app.rag.schemas import RagChatRequest, RagChatResponse
from app.rag.service import RagChatService
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.service import SemanticRetrievalService

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


LLM_PROVIDER_UNAVAILABLE_BODY = {
    "error": {
        "code": "LLM_PROVIDER_UNAVAILABLE",
        "message": "LLM provider unavailable",
    }
}


def llm_provider_unavailable_response() -> JSONResponse:
    return JSONResponse(status_code=502, content=LLM_PROVIDER_UNAVAILABLE_BODY)


@router.post("/chat", response_model=RagChatResponse)
async def chat(
    request: RagChatRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RagChatResponse | JSONResponse:
    settings = get_settings()
    embedding_client = create_embedding_client(settings)
    retrieval_service = SemanticRetrievalService(
        embedding_client=embedding_client,
        repository=ChunkRepository(session),
    )
    service = RagChatService(
        retrieval_service=retrieval_service,
        llm_client=create_llm_client(settings),
        session=session,
    )
    try:
        response = await service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_history=request.include_history,
        )
    except LLMProviderError:
        return llm_provider_unavailable_response()
    await session.commit()
    return response
