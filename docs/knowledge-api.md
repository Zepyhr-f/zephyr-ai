# Knowledge API

Knowledge API 提供知识库只读查询与语义搜索，不返回向量字段，也不调用 LLM。

## Endpoints

```text
GET  /api/v1/knowledge/stats
GET  /api/v1/knowledge/documents?offset=0&limit=50
GET  /api/v1/knowledge/documents/{document_id}
GET  /api/v1/knowledge/chunks/{chunk_id}
POST /api/v1/knowledge/search
```

## Search request

```json
{
  "query": "Zephyr AI 使用什么向量数据库？",
  "top_k": 5
}
```

## Search response

```json
{
  "query": "Zephyr AI 使用什么向量数据库？",
  "top_k": 5,
  "sources": [
    {
      "title": "版本与环境矩阵",
      "file_path": "docs/00-项目简介/01-版本与环境矩阵.md",
      "header_path": "AI 服务运行依赖",
      "score": 0.82,
      "preview": "PostgreSQL / pgvector ...",
      "chunk_index": 1
    }
  ]
}
```

## Safety

- chunk detail 返回 `content`、`header_path`、`metadata`、`preview` 等字段。
- 响应不会返回 `content_vector`。
- `/search` 只执行 embedding + vector search，不生成回答。
