# Conversations / Feedback API

## Conversation read API

```text
GET /api/v1/conversations/{conversation_id}
GET /api/v1/conversations/{conversation_id}/messages?offset=0&limit=50
```

`conversation_id` 是 RAG Chat 返回的外部 `thread_id`，不是内部数据库 UUID。

Messages 按 `created_at` 正序返回。assistant message 会从 `tool_calls.sources` 安全解析 sources；格式缺失或异常时返回空数组。

## Feedback API

```text
POST /api/v1/feedback
GET  /api/v1/feedback/stats
```

### Create feedback

```json
{
  "conversation_id": "thread-uuid-from-rag-chat",
  "message_id": "assistant-message-uuid",
  "rating": "up",
  "comment": "回答有帮助",
  "source": "web",
  "metadata": {
    "page": "chat"
  }
}
```

`rating` 支持：`up`、`down`、`neutral`。

如果提交 `message_id`，服务会校验该 message 存在且属于指定 conversation。

## Safety

Feedback metadata 面向产品侧轻量上下文，不应放入 secret、token、password 或真实连接串。
