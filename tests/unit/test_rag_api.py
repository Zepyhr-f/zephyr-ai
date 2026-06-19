import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.rag import llm_provider_unavailable_response
from app.rag.schemas import RagChatRequest, RagSource
from cli.rag_smoke import print_summary


def test_rag_chat_request_defaults() -> None:
    request = RagChatRequest(message="Zephyr AI 使用什么向量数据库？")

    assert request.conversation_id is None
    assert request.top_k == 5
    assert request.temperature == 0.2
    assert request.max_tokens == 1024
    assert request.include_history is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message", ""),
        ("message", "x" * 4001),
        ("top_k", 0),
        ("top_k", 11),
        ("temperature", -0.1),
        ("temperature", 1.1),
        ("max_tokens", 127),
        ("max_tokens", 4097),
    ],
)
def test_rag_chat_request_validates_bounds(field: str, value) -> None:
    payload = {"message": "hello", field: value}

    with pytest.raises(ValidationError):
        RagChatRequest(**payload)


def test_rag_source_shape() -> None:
    source = RagSource(
        title="版本与环境矩阵",
        file_path="/docs/00-项目简介/01-版本与环境矩阵.md",
        header_path="AI 服务运行依赖",
        score=0.8,
        preview="PostgreSQL / pgvector",
        chunk_index=1,
    )

    assert source.model_dump() == {
        "title": "版本与环境矩阵",
        "file_path": "/docs/00-项目简介/01-版本与环境矩阵.md",
        "header_path": "AI 服务运行依赖",
        "score": 0.8,
        "preview": "PostgreSQL / pgvector",
        "chunk_index": 1,
    }


def test_llm_provider_error_response_shape() -> None:
    response = llm_provider_unavailable_response()

    assert response.status_code == 502
    assert json.loads(response.body) == {
        "error": {
            "code": "LLM_PROVIDER_UNAVAILABLE",
            "message": "LLM provider unavailable",
        }
    }


def test_rag_smoke_prints_preview_without_full_answer(capsys) -> None:
    answer = "A" * 220
    data = {
        "conversation_id": "thread-1",
        "answer": answer,
        "sources": [
            {
                "title": "版本与环境矩阵",
                "file_path": "/docs/00-项目简介/01-版本与环境矩阵.md",
                "header_path": "AI 服务运行依赖",
                "score": 0.8,
            }
        ],
    }

    print_summary(200, data)

    output = capsys.readouterr().out
    assert "status_code=200" in output
    assert "conversation_id=thread-1" in output
    assert "sources_count=1" in output
    assert "answer_preview=" in output
    assert answer not in output
    assert "A" * 160 in output
    assert "top_source_title=版本与环境矩阵" in output
