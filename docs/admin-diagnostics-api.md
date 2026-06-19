# Admin Diagnostics API

Admin API 暴露运行诊断、provider 配置状态和服务版本。响应采用白名单字段，禁止返回 secret、token、password、完整 `DATABASE_URL` 或环境变量 dump。

## Endpoints

```text
GET /api/v1/admin/diagnostics
GET /api/v1/admin/providers
GET /api/v1/admin/version
```

## Providers response

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "ark-code-latest",
    "base_url_host": "ark.cn-beijing.volces.com",
    "configured": true
  },
  "embedding": {
    "provider": "ark",
    "model": "ep-...",
    "base_url_host": "ark.cn-beijing.volces.com",
    "embedding_dim": 2048,
    "configured": true
  }
}
```

## Diagnostics response

包含：

- `service`: name/version。
- `database`: `ok` 或 `unavailable`。
- `knowledge`: document/chunk stats。
- `providers`: 与 `/providers` 相同的脱敏配置状态。

## Testing

```bash
.venv/bin/python -m pytest tests/unit/test_admin_api.py -q
.venv/bin/python -m ruff check app tests cli
```
