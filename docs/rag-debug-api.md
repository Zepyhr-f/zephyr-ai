# RAG Retrieve / Debug API

RAG retrieve/debug API 用于调试检索链路。它复用语义检索，不创建 conversation/message，也不调用 LLM。

## Endpoints

```text
POST /api/v1/rag/retrieve
POST /api/v1/rag/debug
```

## Retrieve request

```json
{
  "query": "Zephyr AI 如何存储向量？",
  "top_k": 5
}
```

## Retrieve response

```json
{
  "query": "Zephyr AI 如何存储向量？",
  "top_k": 5,
  "sources": [
    {
      "title": "架构",
      "file_path": "docs/arch.md",
      "header_path": "数据库",
      "score": 0.8,
      "preview": "Zephyr AI 使用 PostgreSQL / pgvector ...",
      "chunk_index": 1
    }
  ]
}
```

## Debug response fields

`/debug` 额外返回：

- `retrieved_chunks`: 检索到的安全 source 摘要。
- `selected_context`: 将进入 prompt 的上下文文本。
- `prompt_preview`: 非秘密系统提示词 + 用户 query + 检索上下文。
- `estimated_context_chars`: `selected_context` 字符数。

## Safety

`prompt_preview` 不包含 settings、API key、token、password 或完整 `DATABASE_URL`。
