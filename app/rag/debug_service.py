from app.rag.prompt import build_prompt, format_context, source_from_result
from app.rag.schemas import RagDebugResponse, RagRetrieveResponse
from app.retrieval.service import SemanticRetrievalService


class RagDebugService:
    def __init__(self, *, retrieval_service: SemanticRetrievalService) -> None:
        self.retrieval_service = retrieval_service

    async def retrieve(self, *, query: str, top_k: int = 5) -> RagRetrieveResponse:
        normalized_query = query.strip()
        results = await self.retrieval_service.search(normalized_query, limit=top_k)
        return RagRetrieveResponse(
            query=normalized_query,
            top_k=top_k,
            sources=[source_from_result(result) for result in results],
        )

    async def debug(self, *, query: str, top_k: int = 5) -> RagDebugResponse:
        normalized_query = query.strip()
        results = await self.retrieval_service.search(normalized_query, limit=top_k)
        selected_context = "\n\n".join(
            format_context(index, result) for index, result in enumerate(results, start=1)
        )
        prompt_preview = build_prompt(normalized_query, results)
        return RagDebugResponse(
            query=normalized_query,
            top_k=top_k,
            retrieved_chunks=[source_from_result(result) for result in results],
            selected_context=selected_context,
            prompt_preview=prompt_preview,
            estimated_context_chars=len(selected_context),
        )
