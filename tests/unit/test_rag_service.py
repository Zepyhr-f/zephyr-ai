from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.rag.service import RagChatService
from app.retrieval.service import SemanticSearchResult


class FakeRetrievalService:
    def __init__(self, results: list[SemanticSearchResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    async def search(self, query: str, limit: int = 5) -> list[SemanticSearchResult]:
        self.calls.append({"query": query, "limit": limit})
        return self.results


class FakeLLMClient:
    def __init__(self, answer: str = "基于知识库，Zephyr AI 使用 pgvector。") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def chat(self, messages: list, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self.answer


class FakeScalarResult:
    def scalar_one_or_none(self):
        return None


class FakeSession:
    def __init__(self) -> None:
        self.added: list = []
        self.flushed = 0

    async def execute(self, statement):
        return FakeScalarResult()

    def add(self, instance) -> None:
        if getattr(instance, "id", None) is None:
            instance.id = uuid4()
        self.added.append(instance)

    async def flush(self) -> None:
        self.flushed += 1


def make_result() -> SemanticSearchResult:
    return SemanticSearchResult(
        content="Zephyr AI 使用 PostgreSQL / pgvector 存储文档向量。",
        header_path="AI 服务运行依赖",
        document_title="版本与环境矩阵",
        document_file_path="/docs/00-项目简介/01-版本与环境矩阵.md",
        metadata={"chunk_index": 1},
        distance=0.2,
        score=0.8,
    )


@pytest.mark.asyncio
async def test_rag_chat_uses_retrieval_and_llm_and_returns_sources() -> None:
    retrieval = FakeRetrievalService([make_result()])
    llm = FakeLLMClient()
    session = FakeSession()
    service = RagChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        session=session,  # type: ignore[arg-type]
    )

    response = await service.chat(message=" Zephyr AI 使用什么向量数据库？ ", top_k=3)

    assert retrieval.calls == [{"query": "Zephyr AI 使用什么向量数据库？", "limit": 3}]
    assert response.answer == "基于知识库，Zephyr AI 使用 pgvector。"
    assert len(response.sources) == 1
    assert response.sources[0].title == "版本与环境矩阵"
    assert response.sources[0].score == 0.8
    assert "pgvector" in response.sources[0].preview
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][1].content
    assert "知识库上下文" in prompt
    assert "pgvector" in prompt
    assert "Zephyr AI 使用什么向量数据库？" in prompt
    roles = [item.role for item in session.added if hasattr(item, "role")]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_rag_chat_handles_no_sources_without_failing() -> None:
    retrieval = FakeRetrievalService([])
    llm = FakeLLMClient("当前知识库没有足够信息。")
    service = RagChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        session=FakeSession(),  # type: ignore[arg-type]
    )

    response = await service.chat(message="未知问题")

    assert response.sources == []
    prompt = llm.calls[0]["messages"][1].content
    assert "当前没有检索到相关知识库片段" in prompt


@pytest.mark.asyncio
async def test_rag_chat_rejects_blank_message() -> None:
    service = RagChatService(
        retrieval_service=FakeRetrievalService([]),
        llm_client=FakeLLMClient(),
        session=FakeSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="message must not be blank"):
        await service.chat(message="   ")
