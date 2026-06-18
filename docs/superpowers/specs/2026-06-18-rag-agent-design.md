# RAG + Agent 系统设计方案

**日期**：2026-06-18
**项目**：zephyr-ai
**作者**：Claude Code + User

---

## 1. 项目目标

构建一个基于 Python 的 AI 系统，核心能力包括：

- **RAG（检索增强生成）**：读取本地 Markdown 文档，向量化存储，支持语义检索与问答。
- **Agent（智能体）**：基于 LangChain/LangGraph 编排，支持问答和自动化工作流。
- **多 LLM Provider 支持**：OpenAI、Anthropic Claude、Ollama 等可切换。
- **API 服务**：FastAPI 提供 REST + SSE 流式接口。

---

## 2. 架构总览

采用**模块化单体架构**。按功能边界划分为独立模块，在一个代码库中统一管理，职责清晰且避免过早分布式化。

### 2.1 核心数据流（以一次问答为例）

```
用户提问
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   api/      │────▶│ retrieval/  │────▶│ embedding/  │
│  (FastAPI)  │     │  相似度检索  │     │  查询向量化  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  PostgreSQL +       │
                                    │  pgvector           │
                                    └──────────┬──────────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌────▼──────────┐
│   api/      │◀────│   agent/    │◀────│  检索结果+    │
│  返回回答    │     │ LangChain   │     │  上下文组装   │
│  (SSE流式)   │     │ Agent 执行  │     └───────────────┘
└─────────────┘     └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │  LLM   │  │  工具  │  │  工作流 │
         │ Provider│  │ 调用   │  │ 节点   │
         └────────┘  └────────┘  └────────┘
```

### 2.2 关键设计决策

- **统一配置层**：`core/` 管理所有环境变量和 Provider 切换逻辑，避免各模块散落配置。
- **向量化与检索解耦**：`embedding/` 只负责文本→向量，`retrieval/` 只负责向量→文档，方便未来替换模型或索引策略。
- **Agent 层不直接操作数据库**：Agent 通过 `retrieval` 提供的接口获取上下文，保持边界清晰。
- **流式输出**：API 层支持 SSE，Agent 的 LLM 调用也支持流式，用户体验更实时。

---

## 3. 模块设计

### 3.1 模块职责与接口

| 模块 | 职责 | 对外暴露 |
|------|------|----------|
| `core` | 配置管理、LLM Provider 工厂、数据库连接池、日志 | `get_llm(provider: str)`, `get_embeddings(provider: str)`, `get_db()` |
| `ingestion` | MD 文件加载、清洗、分块、元数据提取 | `ingest_document(file_path: Path) -> list[Chunk]` |
| `embedding` | 文本向量化、批量入库、索引管理 | `embed_and_store(chunks: list[Chunk])`, `embed_query(text: str) -> list[float]` |
| `retrieval` | 向量相似度检索、重排序、上下文组装 | `retrieve(query: str, top_k: int = 5) -> list[ContextChunk]` |
| `agent` | LangChain Agent 编排、工具注册、工作流定义 | `run_agent(query: str, thread_id: str, tools: list[str]) -> AsyncIterator[str]` |
| `api` | FastAPI 路由、依赖注入、SSE 流式响应 | `/chat`, `/ingest`, `/documents`, `/health` |
| `cli` | 文档导入、索引重建、查询测试 | `python -m cli ingest ./docs`, `python -m cli query "..."` |

### 3.2 模块边界原则

- `core` 是唯一直接读取环境变量的模块，其他模块通过 `core.config` 获取。
- `agent/` 不直接 import `embedding/`，只通过 `retrieval.retrieve()` 获取上下文。
- `api/` 只做 HTTP 层转换，业务逻辑全部委托给 `agent/` 或 `ingestion/`。

---

## 4. 技术选型

| 层级 | 选型 | 说明 |
|------|------|------|
| Web 框架 | **FastAPI** + `uvicorn` | 原生异步支持、自动 OpenAPI 文档、SSE 流式简单 |
| LLM 交互 | **LangChain** `ChatOpenAI` / `ChatAnthropic` | 统一接口，Provider 切换只需改配置项 |
| Embeddings | **LangChain** `OpenAIEmbeddings` / `HuggingFaceEmbeddings` | 统一接口，支持本地模型降级 |
| 向量数据库 | **PostgreSQL** + **pgvector** | 关系数据与向量同库，事务一致，省掉额外运维 |
| ORM/访问层 | **SQLAlchemy** + **alembic** | 异步支持 (`asyncpg`)，migrations 管理 schema |
| 配置管理 | **Pydantic Settings** | 环境变量自动校验，`.env` 文件支持 |
| 文档解析 | **Markdown** 原生 + `python-frontmatter` | MD 直接用，frontmatter 提取元数据 |
| 分块策略 | **LangChain** `MarkdownHeaderTextSplitter` | 按 Markdown 标题层级分块，保留上下文 |
| Agent 记忆 | **PostgreSQL** 存对话历史 | LangChain `PostgresChatMessageHistory` |
| 工作流引擎 | **LangGraph**（LangChain 官方） | 状态机驱动的多步骤工作流，可视化调试 |

---

## 5. 数据库 Schema

### 5.1 documents 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 文档唯一标识 |
| file_path | VARCHAR | 源文件路径 |
| title | VARCHAR | 文档标题（来自 frontmatter 或文件名） |
| metadata | JSONB | 其他元数据 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 5.2 chunks 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 分块唯一标识 |
| document_id | UUID FK | 关联 documents |
| content | TEXT | 分块文本内容 |
| content_vector | vector(1536) | pgvector 向量（维度取决于 embedding 模型） |
| chunk_index | INT | 在文档中的顺序 |
| header_path | VARCHAR | Markdown 标题路径（如 "# 安装 / ## Docker"） |
| metadata | JSONB | 其他元数据 |
| created_at | TIMESTAMP | 创建时间 |

### 5.3 conversations 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 会话唯一标识 |
| thread_id | VARCHAR | 外部线程 ID |
| title | VARCHAR | 会话标题（可自动生成） |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 5.4 messages 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 消息唯一标识 |
| conversation_id | UUID FK | 关联 conversations |
| role | VARCHAR | user / assistant / system / tool |
| content | TEXT | 消息内容 |
| tool_calls | JSONB | LangChain tool_calls 序列化 |
| created_at | TIMESTAMP | 创建时间 |

---

## 6. API 设计

### 6.1 POST /chat

**请求**：
```json
{
  "query": "如何配置数据库？",
  "thread_id": "uuid-or-user-provided",
  "tools": ["retrieval"],
  "stream": true
}
```

**响应（SSE 流式）**：
```
data: {"type": "token", "content": "首"}
data: {"type": "token", "content": "先"}
...
data: {"type": "done", "sources": [{"doc_id": "...", "title": "...", "score": 0.95}]}
```

### 6.2 POST /ingest

**请求**：
```json
{
  "file_paths": ["./docs/install.md", "./docs/config.md"]
}
```

**响应**：
```json
{
  "ingested": 2,
  "chunks": 15,
  "errors": []
}
```

### 6.3 GET /documents

返回已摄入文档列表，支持分页。

### 6.4 GET /health

健康检查，包含数据库连接状态。

---

## 7. Agent 工作流设计

### 7.1 问答工作流（LangGraph）

```
[开始] ──▶ [检索节点] ──▶ [判断节点]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              [需要工具]            [直接生成]
                    │                   │
                    ▼                   │
              [执行工具]                │
                    │                   │
                    └─────────┬─────────┘
                              ▼
                        [生成回答]
                              │
                              ▼
                          [结束]
```

### 7.2 自动化工作流（预留）

未来可扩展为多步骤工作流，例如：
- 接收任务描述 → 检索相关文档 → 生成执行计划 → 调用工具执行 → 验证结果 → 输出报告

---

## 8. CLI 设计

```bash
# 摄入文档
python -m cli ingest ./docs --recursive

# 重建索引（清空后重新摄入）
python -m cli reindex ./docs

# 本地测试查询
python -m cli query "如何部署？"

# 启动 API 服务
python -m cli serve --host 0.0.0.0 --port 8000
```

---

## 9. 项目目录结构

```
zephyr-ai/
├── .env                          # 环境变量（gitignored）
├── .env.example                  # 环境变量模板
├── pyproject.toml                # Poetry / PEP 621 依赖管理
├── README.md
├── Makefile                      # 常用命令封装
│
├── app/                          # 主应用包
│   ├── __init__.py
│   ├── core/                     # 基础设施层
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic Settings
│   │   ├── database.py           # SQLAlchemy async engine
│   │   ├── llm_factory.py        # get_llm() / get_embeddings()
│   │   └── logging.py            # 结构化日志
│   ├── models/                   # SQLAlchemy ORM + Pydantic Schema
│   │   ├── __init__.py
│   │   ├── base.py               # declarative_base()
│   │   ├── document.py           # documents 表
│   │   ├── chunk.py              # chunks 表（含 vector 字段）
│   │   └── conversation.py       # conversations / messages 表
│   ├── ingestion/                # 文档摄入
│   │   ├── __init__.py
│   │   ├── loader.py             # MD 文件读取 + frontmatter 解析
│   │   ├── splitter.py           # MarkdownHeaderTextSplitter 分块
│   │   └── pipeline.py           # load → split → metadata → Chunk[]
│   ├── embedding/                # 向量化
│   │   ├── __init__.py
│   │   ├── embedder.py           # 调用 embedding API
│   │   └── store.py              # 批量写入 pgvector
│   ├── retrieval/                # 检索
│   │   ├── __init__.py
│   │   ├── search.py             # 向量相似度查询 SQL
│   │   ├── reranker.py           # （预留）重排序
│   │   └── context_builder.py    # 检索结果 → 上下文字符串
│   ├── agent/                    # Agent 编排
│   │   ├── __init__.py
│   │   ├── tools/                # 工具定义
│   │   │   ├── __init__.py
│   │   │   ├── retrieval_tool.py       # retrieve 包装为 @tool
│   │   │   └── workflow_tools.py       # 自定义工作流工具
│   │   ├── graph/                # LangGraph 工作流
│   │   │   ├── __init__.py
│   │   │   ├── nodes.py          # 各工作流节点
│   │   │   └── edges.py          # 状态转移条件
│   │   ├── memory.py             # PostgresChatMessageHistory
│   │   └── runner.py             # run_agent() 统一入口
│   └── api/                      # FastAPI 服务
│       ├── __init__.py
│       ├── main.py               # FastAPI app + 生命周期
│       ├── dependencies.py       # 依赖注入
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── chat.py           # POST /chat (SSE)
│       │   ├── ingest.py         # POST /ingest /reindex
│       │   ├── documents.py      # GET /documents
│       │   └── health.py         # GET /health
│       └── schemas.py            # Pydantic Request/Response
│
├── cli/                          # 命令行工具
│   ├── __init__.py
│   └── main.py                   # Typer CLI
│
├── migrations/                   # alembic 版本
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── tests/                        # 测试
│   ├── conftest.py               # pytest fixtures
│   ├── unit/
│   │   ├── test_ingestion.py
│   │   ├── test_retrieval.py
│   │   └── test_agent.py
│   └── integration/
│       └── test_api.py
│
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-06-18-rag-agent-design.md
│
└── docker-compose.yml            # PostgreSQL + pgvector
```

---

## 10. 环境变量配置

```bash
# LLM 配置
LLM_PROVIDER=openai              # openai | anthropic | ollama
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-7
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Embedding 配置
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# 数据库
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/zephyr_ai

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## 11. 开发环境启动

```bash
# 1. 启动 PostgreSQL + pgvector
docker-compose up -d

# 2. 运行数据库迁移
alembic upgrade head

# 3. 摄入文档
python -m cli ingest ./docs

# 4. 启动 API
python -m cli serve

# 5. 测试查询
python -m cli query "如何配置系统？"
```

---

## 12. 风险与预留

| 风险 | 缓解措施 |
|------|----------|
| 本地模型性能不足 | embedding/llm 支持 fallback 到云 API |
| pgvector 大数量级性能 | 预留 reranker 模块、未来可加 IVF/HNSW 索引优化 |
| Agent 工作流复杂化 | LangGraph 状态机可视化调试，逐步迭代 |
| 多文档重复摄入 | documents 表用 file_path + hash 去重 |

---

## 13. 后续迭代方向

1. **文档同步机制**：从 GitHub/S3/Notion 自动拉取 MD 文档。
2. **多模态支持**：扩展至图片、PDF 等非文本内容。
3. **多 Agent 协作**：使用 `dispatching-parallel-agents` 并行处理独立子任务。
4. **权限与隔离**：多租户场景下文档访问控制。
5. ** observability**：集成 LangSmith 追踪 Agent 执行链路。
