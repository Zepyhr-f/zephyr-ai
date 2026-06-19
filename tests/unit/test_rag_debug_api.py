import pytest

from app.rag.debug_service import RagDebugService
from app.retrieval.service import SemanticSearchResult


class FakeRetrievalService:
    def __init__(self):
        self.calls = []

    async def search(self, query, limit=5):
        self.calls.append({"query": query, "limit": limit})
        return [
            SemanticSearchResult(
                content="Zephyr AI 使用 pgvector。",
                header_path="数据库",
                document_title="架构",
                document_file_path="docs/arch.md",
                metadata={"chunk_index": 1},
                distance=0.2,
                score=0.8,
            )
        ]


@pytest.mark.asyncio
async def test_retrieve_only_runs_retrieval() -> None:
    retrieval = FakeRetrievalService()
    service = RagDebugService(retrieval_service=retrieval)

    response = await service.retrieve(query=" pgvector ", top_k=4)

    assert retrieval.calls == [{"query": "pgvector", "limit": 4}]
    assert response.sources[0].title == "架构"
    assert response.sources[0].chunk_index == 1


@pytest.mark.asyncio
async def test_debug_returns_prompt_preview_without_secrets() -> None:
    service = RagDebugService(retrieval_service=FakeRetrievalService())

    response = await service.debug(query="如何存储向量？", top_k=1)
    dumped = response.model_dump_json()

    assert "知识库上下文" in response.prompt_preview
    assert "Zephyr AI 使用 pgvector" in response.selected_context
    assert response.estimated_context_chars == len(response.selected_context)
    assert "secret" not in dumped.lower()
    assert "password" not in dumped.lower()
    assert "token" not in dumped.lower()
