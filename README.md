# Zephyr AI

Zephyr AI 是 Zephyr 项目的 AI 能力中心，当前主要提供基于项目文档的 RAG 问答能力，并为后续 Agent、运维助手、知识库管理后台等能力预留扩展空间。

当前生产知识库来源于：

```text
/home/ubuntu/workspace/zephyr-docs
```

核心链路：

```text
Markdown 文档
  -> 文档解析与切分
  -> Embedding 向量生成
  -> PostgreSQL / pgvector 存储
  -> 语义检索
  -> LLM 生成回答
  -> RAG Chat API
```

---

## 1. 项目定位

Zephyr AI 当前承担三类职责：

1. **RAG 知识库服务**
   读取 Zephyr 文档，生成向量并存入 pgvector，对外提供语义检索与 RAG Chat API。

2. **Zephyr 前端 AI 能力后端**
   `zephyr-blog` 通过 Nginx 同源代理调用 Zephyr AI，提供博客左下角 AI 聊天入口。

3. **未来 Agent 能力基础**
   后续可扩展为只读运维助手、项目文档助手、代码理解助手、管理后台 AI Copilot。

---

## 2. 当前能力

已实现能力：

- FastAPI HTTP 服务
- `/health` 健康检查
- Markdown 文档读取与元数据解析
- 文档 chunk 切分
- Ark multimodal embedding 调用
- 2048 维 pgvector 向量存储
- 语义检索
- RAG Chat API
- 多轮会话 `conversation_id`
- 最近历史对话注入
- sources 引用来源返回
- LLM provider 异常统一映射
- CLI smoke 工具
- Docker Compose 部署

主要 API：

```text
GET  /health
POST /api/v1/rag/chat
```

---

## 3. 技术栈

| 类型 | 技术 |
| --- | --- |
| 语言 | Python 3.11 |
| Web 框架 | FastAPI |
| ASGI Server | Uvicorn |
| 配置 | pydantic-settings |
| ORM | SQLAlchemy asyncio |
| 数据库驱动 | asyncpg |
| 迁移 | Alembic |
| 向量库 | PostgreSQL + pgvector |
| Embedding | 火山方舟 Ark multimodal embeddings |
| LLM | Anthropic-compatible provider / OpenAI-compatible provider |
| RAG 相关 | LangChain / LangGraph 预留 |
| 测试 | pytest / pytest-asyncio |
| 代码规范 | black / ruff |
| 部署 | Docker / Docker Compose |

---

## 4. 目录结构

```text
zephyr-ai/
├── app/
│   ├── api/                  # FastAPI 路由
│   ├── core/                 # 配置、数据库、Embedding/LLM client
│   ├── ingestion/            # 文档读取、切分、入库
│   ├── models/               # SQLAlchemy 模型
│   ├── rag/                  # RAG Chat schema 与业务服务
│   ├── repositories/         # 数据访问层
│   ├── retrieval/            # 语义检索服务
│   └── utils/                # 向量归一化等工具
├── cli/
│   ├── ingest_docs.py        # 文档入库 CLI
│   └── rag_smoke.py          # RAG API smoke CLI
├── docs/
│   └── rag-chat-api.md       # RAG Chat API 文档
├── migrations/               # Alembic migrations
├── tests/                    # 单元测试
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 5. 环境变量

请从 `.env.example` 复制本地配置：

```bash
cp .env.example .env
```

> 注意：真实 key、token、password、数据库连接串只允许写入本地 `.env`，不要提交到 Git，不要写入文档、日志或 prompt。

关键配置示例：

```bash
# LLM
LLM_PROVIDER=anthropic
ANTHROPIC_BASE_URL=https://api.example.com/anthropic
ANTHROPIC_MODEL=your-model
ANTHROPIC_AUTH_TOKEN=[REDACTED]

# Embedding
EMBEDDING_PROVIDER=ark
ARK_EMBEDDING_API_KEY=[REDACTED]
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_ENDPOINT=/embeddings/multimodal
ARK_EMBEDDING_MODEL=your_embedding_endpoint_id
EMBEDDING_DIMENSION=2048

# PostgreSQL / pgvector
DATABASE_URL=postgresql+asyncpg://zephyr:[REDACTED]@pgsql-pgvector:5432/zephyr_ai

# API
API_HOST=0.0.0.0
API_PORT=8000
ZEPHYR_AI_PORT=8000
LOG_LEVEL=INFO
```

当前 embedding 主线使用 Ark multimodal endpoint：

```text
POST /api/v3/embeddings/multimodal
```

文本 input 格式：

```json
{"type":"text","text":"..."}
```

向量维度：

```text
2048
```

---

## 6. 数据库与基础设施

Zephyr AI 默认复用 `zephyr-infra` 中已经存在的 PostgreSQL / pgvector 服务。

Docker 网络：

```text
infra_default
```

PostgreSQL 容器：

```text
pgsql-pgvector
```

数据库建议：

```text
zephyr_ai
```

部署前请确保基础设施已启动：

```bash
cd /home/ubuntu/workspace
./run.sh infra start
```

或检查状态：

```bash
cd /home/ubuntu/workspace
./run.sh status
```

---

## 7. 本地开发

安装依赖：

```bash
cd /home/ubuntu/workspace/zephyr-ai
poetry install
```

运行迁移：

```bash
poetry run alembic upgrade head
```

启动服务：

```bash
poetry run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

预期：

```json
{"status":"ok","service":"zephyr-ai","database":"ok"}
```

---

## 8. Docker 部署

Zephyr AI 可以通过项目内 Docker Compose 启动：

```bash
cd /home/ubuntu/workspace/zephyr-ai
docker compose up -d --build
```

也可以通过 workspace 统一脚本管理：

```bash
cd /home/ubuntu/workspace
./run.sh ai rebuild
./run.sh verify
```

容器端口默认只绑定本机：

```text
127.0.0.1:8000 -> zephyr-ai:8000
```

博客前端通过 Nginx 同源代理访问：

```text
/ai-api/ -> zephyr-ai:8000
```

博客侧实际调用地址：

```text
/ai-api/api/v1/rag/chat
```

---

## 9. 文档入库

当前生产文档来源：

```text
/home/ubuntu/workspace/zephyr-docs
```

文档入库 CLI：

```bash
cd /home/ubuntu/workspace/zephyr-ai
poetry run python cli/ingest_docs.py \
  --docs-root /home/ubuntu/workspace/zephyr-docs
```

如果在 Docker 容器内执行，请确保容器能访问文档目录，或在部署层挂载对应目录。

入库流程：

```text
读取 Markdown
  -> 解析 frontmatter / 标题 / 文件路径
  -> 按标题和长度切分 chunk
  -> 调用 embedding provider
  -> L2 normalize
  -> 写入 documents / chunks 表
  -> chunks.content_vector 存储 2048 维向量
```

---

## 10. RAG Chat API

接口地址：

```text
POST /api/v1/rag/chat
```

请求示例：

```bash
curl -s http://127.0.0.1:8000/api/v1/rag/chat \
  -H 'content-type: application/json' \
  -d '{
    "message": "Zephyr AI 使用什么向量数据库？",
    "top_k": 5,
    "temperature": 0.2,
    "max_tokens": 1024,
    "include_history": true
  }'
```

响应示例：

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

多轮对话时，把上一次响应里的 `conversation_id` 传回即可：

```bash
curl -s http://127.0.0.1:8000/api/v1/rag/chat \
  -H 'content-type: application/json' \
  -d '{
    "conversation_id": "***",
    "message": "它的入库流程是什么？",
    "include_history": true
  }'
```

更多说明见：

```text
docs/rag-chat-api.md
```

---

## 11. Smoke 验证

RAG smoke：

```bash
cd /home/ubuntu/workspace/zephyr-ai
poetry run python cli/rag_smoke.py \
  --url http://127.0.0.1:8000/api/v1/rag/chat \
  --message 'Zephyr AI 使用什么向量数据库？' \
  --top-k 5
```

该工具只输出：

```text
status_code
conversation_id
answer_preview
sources_count
top source 的 title/file/header/score
```

它不会输出完整 answer、完整 prompt、完整 context、完整向量或任何密钥。

---

## 12. 测试

运行单元测试：

```bash
cd /home/ubuntu/workspace/zephyr-ai
poetry run pytest
```

运行指定测试：

```bash
poetry run pytest tests/unit/test_rag_api.py
poetry run pytest tests/unit/test_rag_service.py
poetry run pytest tests/unit/test_semantic_retrieval.py
```

代码格式化与检查：

```bash
poetry run black app cli tests
poetry run ruff check app cli tests
```

---

## 13. 与 Zephyr Blog 的关系

`zephyr-blog` 已接入 Zephyr AI：

```text
用户打开 blog.zephyr.green
  -> 左下角 AI 聊天入口
  -> 前端请求 /ai-api/api/v1/rag/chat
  -> Nginx 代理到 zephyr-ai:8000
  -> Zephyr AI 执行 RAG
  -> 返回 answer + sources
```

Nginx 代理位于：

```text
/home/ubuntu/workspace/zephyr-infra/nginx/conf.d/blog.conf
```

核心代理规则：

```text
/ai-api/ -> zephyr-ai:8000
```

---

## 14. 安全约定

必须遵守：

- 不提交 `.env`
- 不提交真实 API key
- 不提交真实 token
- 不提交数据库密码
- 不在 README、docs、prompt、日志中输出真实密钥
- 不输出完整 prompt/context/vector
- smoke 工具只输出摘要信息
- 本地部署适配优先放入 ignored override 文件

敏感信息统一写成：

```text
[REDACTED]
```

---

## 15. 后续规划

建议后续阶段：

1. **RAG 知识库管理能力**
   - 查看 documents / chunks
   - 查看入库状态
   - 手动触发重新入库
   - 检索效果测试

2. **Zephyr AI 文档补全**
   - 架构说明
   - RAG 数据流
   - Embedding 生成流程
   - pgvector 存储说明
   - 部署与运维说明

3. **博客 AI 体验优化**
   - 当前页面上下文注入
   - sources 展示优化
   - 停止生成 / 重新生成
   - 复制回答
   - 聊天窗口位置与宽度持久化

4. **只读运维 Agent**
   - 容器状态检查
   - 健康检查
   - 日志摘要
   - RAG 状态诊断

5. **代码理解助手**
   - 索引 Zephyr 多仓库代码
   - 回答模块职责、接口链路、调用关系
   - 生成变更摘要与排查建议

---

## 16. 常用命令速查

```bash
# 查看服务状态
cd /home/ubuntu/workspace
./run.sh status

# 重建 Zephyr AI
./run.sh ai rebuild

# 全局验证
./run.sh verify

# 查看 Zephyr AI 日志
./run.sh logs ai

# 本地健康检查
curl -s http://127.0.0.1:8000/health

# RAG smoke
cd /home/ubuntu/workspace/zephyr-ai
poetry run python cli/rag_smoke.py \
  --url http://127.0.0.1:8000/api/v1/rag/chat \
  --message 'Zephyr AI 使用什么向量数据库？' \
  --top-k 5
```
