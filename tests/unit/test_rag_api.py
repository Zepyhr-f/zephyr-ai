from app.rag.schemas import RagChatRequest, RagSource


def test_rag_chat_request_defaults() -> None:
    request = RagChatRequest(message="Zephyr AI 使用什么向量数据库？")

    assert request.conversation_id is None
    assert request.top_k == 5


def test_rag_chat_request_validates_top_k_upper_bound() -> None:
    try:
        RagChatRequest(message="hello", top_k=11)
    except Exception as exc:
        assert "less than or equal to 10" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rag_source_shape() -> None:
    source = RagSource(
        title="版本与环境矩阵",
        file_path="/docs/00-项目简介/01-版本与环境矩阵.md",
        header_path="AI 服务运行依赖",
        score=0.8,
        preview="PostgreSQL / pgvector",
    )

    assert source.model_dump() == {
        "title": "版本与环境矩阵",
        "file_path": "/docs/00-项目简介/01-版本与环境矩阵.md",
        "header_path": "AI 服务运行依赖",
        "score": 0.8,
        "preview": "PostgreSQL / pgvector",
    }
