from app.knowledge.schemas import KnowledgeChunkDetail, KnowledgeSearchRequest


def test_knowledge_search_request_bounds() -> None:
    request = KnowledgeSearchRequest(query="pgvector")

    assert request.top_k == 5


def test_knowledge_chunk_schema_excludes_vector() -> None:
    response = KnowledgeChunkDetail(
        id="chunk-1",
        document_id="doc-1",
        document_title="架构",
        document_file_path="docs/arch.md",
        content="内容",
        header_path="标题",
        metadata={"chunk_index": 1},
        chunk_index=1,
        preview="内容",
        created_at="2026-06-19T00:00:00Z",
        updated_at="2026-06-19T00:00:00Z",
    )

    assert "content_vector" not in response.model_dump()
