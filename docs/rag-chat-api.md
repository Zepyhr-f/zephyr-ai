# RAG Chat API

## API 地址

`POST /api/v1/rag/chat`

该接口基于 Zephyr 文档知识库进行语义检索，并调用 LLM 生成回答。当前生产知识库来自 `/home/ubuntu/workspace/zephyr-docs`，本地数据库已入库多个文档 chunks。

## 请求字段

```json
{
  "message": "Zephyr AI 使用什么向量数据库？",
  "conversation_id": "***",
  "top_k": 5,
  "temperature": 0.2,
  "max_tokens": 1024,
  "include_history": true
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `message` | string | 是 | - | 用户问题，长度 1~4000。 |
| `conversation_id` | string/null | 否 | `null` | 会话 ID。存在时复用会话，不存在时创建新会话。 |
| `top_k` | integer | 否 | `5` | 检索返回片段数，范围 1~10。 |
| `temperature` | number | 否 | `0.2` | 传给 LLM provider 的温度参数，范围 0~1。 |
| `max_tokens` | integer | 否 | `1024` | 传给 LLM provider 的最大输出 token 数，范围 128~4096。 |
| `include_history` | boolean | 否 | `true` | 当 `conversation_id` 存在时，是否把最近历史对话加入 prompt。 |

## 响应字段

```json
{
  "conversation_id": "***",
  "answer": "基于知识库的回答...",
  "sources": [
    {
      "title": "版本与环境矩阵",
      "file_path": "/docs/00-项目简介/01-版本与环境矩阵.md",
      "header_path": "AI 服务运行依赖",
      "score": 0.8,
      "preview": "PostgreSQL / pgvector...",
      "chunk_index": 1
    }
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `conversation_id` | string | 会话 ID，后续多轮请求可继续传入。 |
| `answer` | string | LLM 生成的回答。 |
| `sources` | array | 本次回答使用的检索来源。 |

## sources 说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `title` | string/null | 文档标题。 |
| `file_path` | string | 文档路径。 |
| `header_path` | string/null | chunk 所属标题路径。 |
| `score` | number | 语义检索相似度分数。 |
| `preview` | string | chunk 内容预览，接口只返回短预览。 |
| `chunk_index` | integer/null | chunk 序号，来自检索结果 `metadata["chunk_index"]`；无法转换时为 `null`。 |

## 多轮示例

第一次请求：

```bash
curl -s http://localhost:8000/api/v1/rag/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Zephyr AI 使用什么向量数据库？"}'
```

第二次请求复用返回的 `conversation_id`：

```bash
curl -s http://localhost:8000/api/v1/rag/chat \
  -H 'content-type: application/json' \
  -d '{"conversation_id":"***","message":"它的入库流程是什么？","include_history":true}'
```

当 `include_history=true` 且 `conversation_id` 存在时，服务会读取该会话最近 6 条 messages，并在 prompt 中加入“历史对话”。单条历史内容会截断到约 800 字，避免 prompt 过长。

如果不希望注入历史上下文：

```bash
curl -s http://localhost:8000/api/v1/rag/chat \
  -H 'content-type: application/json' \
  -d '{"conversation_id":"***","message":"只基于当前问题回答","include_history":false}'
```

无论是否注入历史，本次 user/assistant 消息都会继续写入该会话。

## 错误码

Provider 不可用时返回 HTTP 502：

```json
{
  "error": {
    "code": "LLM_PROVIDER_UNAVAILABLE",
    "message": "LLM provider unavailable"
  }
}
```

请求字段校验错误使用 FastAPI 默认 422 响应。

## Smoke 命令

```bash
python3 cli/rag_smoke.py \
  --url http://localhost:8000/api/v1/rag/chat \
  --message 'Zephyr AI 使用什么向量数据库？' \
  --top-k 5
```

继续同一会话：

```bash
python3 cli/rag_smoke.py \
  --url http://localhost:8000/api/v1/rag/chat \
  --conversation-id '***' \
  --message '它的入库流程是什么？' \
  --top-k 5
```

`rag_smoke.py` 只输出：`status_code`、`conversation_id`、`answer_preview`、`sources_count`、top source 的 `title/file/header/score`。它不会输出完整 answer、完整 prompt、完整 context、完整向量或任何 key。

## 环境变量说明

RAG Chat 依赖数据库、Embedding provider 和 LLM provider 配置。示例值中的 key/token 均使用 `***`：

```bash
DATABASE_URL=postgresql+asyncpg://postgres:***@localhost:5432/zephyr_ai
LLM_PROVIDER=anthropic
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_AUTH_TOKEN=***
ANTHROPIC_MODEL=***
ARK_API_KEY=***
ARK_BASE_URL=***
ARK_EMBEDDING_MODEL=***
```

不要把真实 key/token 写入文档、日志、prompt 或 smoke 输出。
