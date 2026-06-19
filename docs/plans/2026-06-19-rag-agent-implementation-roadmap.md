# Zephyr AI RAG + Agent 全局路线计划

**日期**：2026-06-19  
**项目**：`zephyr-ai`  
**定位**：Zephyr 项目的 AI 能力中心，先建设项目知识库 RAG，再扩展为可调用工具、可记忆、可编排任务的 Agent 服务。

---

## 1. 总目标

`zephyr-ai` 要最终提供三层能力：

1. **Knowledge RAG**
   - 摄入 `zephyr-docs`、部署文档、项目说明、接口文档等 Markdown 文档。
   - 将文档切分、向量化、写入 PostgreSQL/pgvector。
   - 支持基于语义检索的项目知识问答，并返回引用来源。

2. **Project Agent**
   - 在 RAG 基础上加入工具调用和任务流程编排。
   - 支持“先查知识库 → 规划步骤 → 调用工具/服务 → 总结结果”的工作流。
   - 先做轻量 Agent Runner，后续再决定是否引入 LangGraph 复杂编排。

3. **App AI Service**
   - 对外提供 HTTP API，供 `zephyr-admin`、`zephyr-blog` 或后续业务系统调用。
   - 支持普通 JSON 响应和 SSE 流式响应。
   - 支持会话记忆、消息持久化、来源追踪和运行审计。

---

## 2. 当前基础状态

当前 `zephyr-ai` 已有：

- FastAPI 服务基础。
- `/health` 健康检查。
- Docker Compose 启动能力。
- 复用现有 `zephyr-infra` 的 PostgreSQL/pgvector。
- 独立数据库：`zephyr_ai`。
- SQLAlchemy async 数据库连接。
- Alembic 迁移。
- RAG/Agent 基础表：
  - `documents`
  - `chunks`
  - `conversations`
  - `messages`
- `chunks.content_vector vector(1536)` pgvector 字段。

当前还缺少：

- LLM / Embedding 客户端抽象。
- Markdown 摄入。
- 文档切分。
- Embedding 生成。
- 向量入库。
- 语义检索。
- RAG Chat API。
- Agent Runner。
- CLI ingest/query/reindex。
- `zephyr-docs` 端到端验收场景。

---

## 3. 全局设计原则

### 3.1 哈宝负责全局路线，Claude Code 负责阶段实现

- 本文档是全局路线图，由哈宝维护。
- Claude Code 后续只负责某个阶段内部的代码实现。
- Claude Code 不负责修改整体路线、产品定位和阶段优先级。
- 每个阶段开始前，由哈宝明确阶段目标、验收标准和边界，再交给 Claude Code 编码。

### 3.2 敏感信息边界

项目是开源项目，因此：

- 真实 Token、API Key、数据库密码、私钥不能进入 Git/GitHub。
- `.env.example`、文档、测试样例只能使用占位符。
- 真实值只允许存在：
  - 服务器本地 `.env`
  - shell 环境变量
  - 被 `.gitignore` 忽略的本地 override 文件

文档中统一使用：

```text
replace-me
<api-key>
<token>
<password>
```

### 3.3 优先轻量实现，再逐步增强

- 第一阶段不急着把 LangChain/LangGraph 全部接入运行镜像。
- Provider 调用优先使用轻量 `httpx` 客户端抽象。
- 等 RAG 主链路跑通后，再决定哪些地方引入 LangChain/LangGraph。

### 3.4 全链路异步

以下部分统一使用 async：

- FastAPI API
- 数据库访问
- Provider 调用
- Embedding 调用
- 检索
- Agent Runner
- SSE 流式输出

### 3.5 每阶段必须可测试、可验证

每个阶段都要有：

- 单元测试。
- 必要的集成测试。
- 明确验收命令。
- 明确成功标准。
- 若依赖真实 Provider，测试必须可跳过或使用 fake client。

---

## 4. 模型与 Provider 规划

### 4.1 对话 / Agent 模型

本地运行时使用 Anthropic 兼容接口：

```env
ANTHROPIC_AUTH_TOKEN=<token>
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding
ANTHROPIC_MODEL=ark-code-latest
```

说明：

- 真实 token 只写入本地 `.env`。
- 文档和 `.env.example` 不写真实 token。
- 业务代码通过配置读取，不硬编码。

### 4.2 Embedding 模型

Embedding 使用 Ark/OpenAI 兼容接口：

```env
EMBEDDING_PROVIDER=ark
ARK_EMBEDDING_API_KEY=<api-key>
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding
EMBEDDING_DIMENSION=1536
```

说明：

- 初期只测试这两个模型链路：
  - 对话 / Agent：`ark-code-latest`
  - Embedding：`doubao-embedding`
- 如果模型调用不通或返回格式不兼容，该阶段停止并报告原因。

---

## 5. 阶段路线图

## 阶段 1：Provider 配置与客户端抽象

### 目标

补齐 LLM 和 Embedding 的配置与轻量客户端，为后续 RAG/Agent 提供统一调用入口。

### 主要能力

- 读取 Anthropic 兼容对话配置。
- 读取 Ark/OpenAI 兼容 Embedding 配置。
- 提供统一 `LLMClient`。
- 提供统一 `EmbeddingClient`。
- 支持 fake client 单元测试，不依赖真实网络。

### 预计文件

```text
app/core/config.py
app/core/llm_client.py
app/core/embedding_client.py
.env.example
Dockerfile
tests/unit/test_llm_clients.py
```

### 验收标准

- 配置能从环境变量读取。
- LLM client 能构造正确请求。
- Embedding client 能构造正确请求。
- 测试不泄漏 token。
- `/health` 不被破坏。

### 建议验证命令

```bash
python3 -m py_compile app/core/config.py app/core/llm_client.py app/core/embedding_client.py
pytest tests/unit/test_llm_clients.py -q
curl http://127.0.0.1:8000/health
```

---

## 阶段 2：Markdown 文档摄入与切分

### 目标

实现把 Markdown 文档读取成稳定、可追踪、可向量化的 chunk。

### 主要能力

- 读取 `.md` 文件。
- 支持 frontmatter。
- 提取标题、路径、文件 hash、更新时间等元数据。
- 按 Markdown 标题优先切分。
- 大段内容按长度 fallback 切分。
- chunk 结果稳定，便于重复摄入时去重。

### 预计文件

```text
app/ingestion/__init__.py
app/ingestion/loader.py
app/ingestion/splitter.py
app/ingestion/pipeline.py
tests/unit/test_ingestion.py
```

### 验收标准

- 一个 Markdown 文件能切出稳定 chunks。
- 每个 chunk 包含：
  - content
  - source_path
  - title
  - header_path
  - chunk_index
  - content_hash
- 不支持的文件类型有明确错误。

---

## 阶段 3：Embedding 生成与 pgvector 存储

### 目标

把文档 chunk 调用 Embedding 模型生成向量，并写入 PostgreSQL/pgvector。

### 主要能力

- 批量生成 embedding。
- 写入 `documents`。
- 写入 `chunks`。
- 支持重复摄入时 upsert，不重复插入。
- 支持内容变化时更新 chunk 和 vector。

### 预计文件

```text
app/embedding/__init__.py
app/embedding/service.py
app/embedding/store.py
app/repositories/documents.py
migrations/versions/*.py  # 仅当现有 schema 不足时新增
tests/unit/test_embedding.py
```

### 验收标准

- fake embedding 可完成入库测试。
- 真实 pgvector 可完成 round-trip 验证。
- 重复 ingest 不产生重复 chunks。
- 真实 key 不进入测试输出。

---

## 阶段 4：语义检索与上下文组装

### 目标

实现用户问题 → query embedding → pgvector top-k 检索 → 上下文组装。

### 主要能力

- query embedding。
- pgvector 相似度检索。
- top-k 返回。
- 支持 source/path 过滤。
- 组装 RAG context。
- 返回 sources 用于回答引用。

### 预计文件

```text
app/retrieval/__init__.py
app/retrieval/search.py
app/retrieval/context_builder.py
tests/unit/test_retrieval.py
```

### 验收标准

- 能返回 top-k chunks。
- 返回 score、document_id、chunk_id、source_path、header_path。
- context builder 能控制最大长度。
- 无检索结果时有安全 fallback。

---

## 阶段 5：RAG Chat API

### 目标

提供可用的 RAG 问答接口。

### 主要能力

- `POST /chat` 非流式问答。
- 可选 SSE 流式输出。
- 自动检索上下文。
- 构造 RAG Prompt。
- 调用 LLM 生成回答。
- 返回 sources。
- Provider 错误时返回安全错误，不泄漏密钥。

### 预计文件

```text
app/api/schemas.py
app/api/routers/chat.py
app/api/main.py
tests/unit/test_chat_api.py
tests/integration/test_api.py
```

### API 草案

```http
POST /chat
```

请求：

```json
{
  "query": "Zephyr 怎么部署？",
  "thread_id": "optional-thread-id",
  "top_k": 5,
  "stream": false
}
```

响应：

```json
{
  "answer": "...",
  "sources": [
    {
      "document_id": "...",
      "chunk_id": "...",
      "source_path": "...",
      "title": "...",
      "score": 0.82
    }
  ],
  "thread_id": "..."
}
```

### 验收标准

- fake LLM 下 API 测试通过。
- 真实配置下能对已摄入文档问答。
- 回答必须带 sources。

---

## 阶段 6：Agent 工具封装与基础 Agent Runner

### 目标

在 RAG 能力基础上加入 Agent 工具调用能力。

### 主要能力

- 把 retrieval 封装为 tool。
- Agent Runner 能决定是否调用 retrieval。
- Agent 输出包含 tool trace。
- 初期先做轻量 runner，不强依赖 LangGraph。
- 后续复杂工作流再引入 LangGraph。

### 预计文件

```text
app/agent/__init__.py
app/agent/tools/retrieval_tool.py
app/agent/runner.py
tests/unit/test_agent.py
```

### 验收标准

- Agent 能调用 retrieval tool。
- Agent 能基于 tool 返回的 context 生成回答。
- tool trace 可用于调试。
- 无工具场景也能直接回答。

---

## 阶段 7：对话记忆与消息持久化

### 目标

让 Chat/Agent 支持会话上下文和消息落库。

### 主要能力

- 创建或读取 conversation。
- 保存 user message。
- 保存 assistant message。
- 保存 tool_calls / tool_trace。
- 加载最近 N 条消息作为上下文。

### 预计文件

```text
app/repositories/conversations.py
app/agent/memory.py
app/models/conversation.py  # 仅当现有 schema 不足时修改
migrations/versions/*.py    # 仅当 schema 变化时新增
tests/unit/test_memory.py
```

### 验收标准

- 同一 `thread_id` 可连续对话。
- 历史消息按时间顺序加载。
- 消息数量有上限，避免 prompt 无限膨胀。

---

## 阶段 8：CLI 工具

### 目标

提供本地运维和开发友好的 CLI。

### 主要能力

- ingest 文档。
- reindex 文档。
- query 本地知识库。
- serve API。
- 查看文档/块统计。

### 预计文件

```text
cli/main.py
cli/__init__.py
tests/unit/test_cli.py
```

### CLI 草案

```bash
python -m cli.main ingest /home/ubuntu/workspace/zephyr-docs
python -m cli.main query "Zephyr AI 使用哪个数据库？"
python -m cli.main reindex /home/ubuntu/workspace/zephyr-docs --force
python -m cli.main serve --host 0.0.0.0 --port 8000
```

### 验收标准

- CLI 参数解析正确。
- 错误信息清晰。
- CLI 不打印敏感环境变量。
- 能摄入 `zephyr-docs`。

---

## 阶段 9：Docker 与部署验证

### 目标

保证 RAG/Agent 功能进入容器后仍可运行。

### 主要能力

- Docker 镜像包含运行依赖。
- 继续复用 `pgsql-pgvector`。
- `docker compose up -d --build` 可启动。
- `alembic upgrade head` 可执行。
- `run.sh ai restart` 可用。
- `run.sh verify` 包含 AI 健康检查。

### 预计文件

```text
Dockerfile
docker-compose.yml
.dockerignore
.env.example
/home/ubuntu/workspace/run.sh  # workspace 本地脚本，不属于 zephyr-ai Git 仓库
```

### 验收标准

```bash
cd /home/ubuntu/workspace/zephyr-ai
docker compose up -d --build
docker compose exec -T zephyr-ai alembic upgrade head
curl http://127.0.0.1:8000/health

cd /home/ubuntu/workspace
./run.sh ai restart
./run.sh verify
```

预期：

```text
zephyr-ai health: 200
verification passed
```

---

## 阶段 10：`zephyr-docs` 知识库验收场景

### 目标

用真实项目文档验证 RAG 是否有业务价值。

### 验收语料

```text
/home/ubuntu/workspace/zephyr-docs
```

### 验收问题示例

- Zephyr 的部署目录怎么规划？
- zephyr-ai 使用哪个数据库？
- 如何重启 gateway？
- 前端域名的 API 是怎么反代的？
- 当前为什么 HTTPS 暂不处理？
- PostgreSQL、Redis、NATS、MinIO 分别是什么作用？

### 验收标准

- 成功摄入 `zephyr-docs`。
- 文档数、chunk 数非零。
- 每个问题能返回答案。
- 每个答案至少包含一个 source。
- 答案内容能对应到真实文档。
- 如果检索不到，应明确说明不知道，不编造。

---

## 6. 推荐执行顺序

优先级从高到低：

1. 阶段 1：Provider 配置与客户端抽象。
2. 阶段 2：Markdown 文档摄入与切分。
3. 阶段 3：Embedding 生成与 pgvector 存储。
4. 阶段 4：语义检索与上下文组装。
5. 阶段 5：RAG Chat API。
6. 阶段 10：`zephyr-docs` 知识库验收。
7. 阶段 7：对话记忆与消息持久化。
8. 阶段 8：CLI 工具。
9. 阶段 6：Agent 工具封装与基础 Agent Runner。
10. 阶段 9：Docker 与部署验证贯穿每个阶段持续维护。

说明：

- RAG 主链路优先于 Agent。
- Agent 必须建立在稳定 retrieval 和 chat 基础上。
- Docker 验证每阶段都要做，避免最后集中爆雷。

---

## 7. 每阶段执行规则

每个阶段执行时统一遵守：

1. 哈宝先明确阶段目标和验收标准。
2. Claude Code 负责该阶段内部代码实现。
3. Claude Code 默认使用：
   ```bash
   --dangerously-skip-permissions
   ```
4. 哈宝负责：
   - 审查 diff。
   - 跑测试。
   - 跑 Docker/health 验证。
   - 检查敏感信息。
   - 判断是否进入下一阶段。
5. 每阶段完成后再考虑是否提交 Git。
6. 提交前必须确认：
   - `.env` 未提交。
   - token/key/password 未提交。
   - 本地 `docker-compose.override.yml` 未提交。
   - 运行数据、缓存、日志未提交。

---

## 8. 近期最小 MVP

最小 MVP 定义为：

1. 能摄入 `zephyr-docs` Markdown。
2. 能生成 embedding 并写入 pgvector。
3. 能通过语义检索找到相关 chunks。
4. 能通过 `/chat` 回答问题。
5. 回答带 sources。
6. 能通过 Docker 运行。
7. 能用 `./run.sh ai restart` 和 `./run.sh verify` 验证。

MVP 不包含：

- 复杂多工具 Agent。
- 多 Agent 协作。
- 权限系统。
- 多租户。
- UI 管理页面。
- LangGraph 复杂状态机。

这些放到 MVP 后续阶段。

---

## 9. 风险与处理策略

| 风险 | 影响 | 处理策略 |
|---|---|---|
| Ark/Anthropic 兼容接口格式差异 | Chat/Agent 调用失败 | 阶段 1 先用可注入 fake client 测 payload；真实调用单独 smoke test |
| `doubao-embedding` 维度与默认 1536 不一致 | pgvector 写入失败 | 阶段 1/3 必须验证实际返回维度；必要时新增迁移或调整配置 |
| Docker 镜像依赖过重 | 构建慢或失败 | 先轻量 httpx 实现；LangChain/LangGraph 后置 |
| 文档重复摄入 | 数据膨胀、检索污染 | 使用 source_path + content_hash 做幂等 upsert |
| 回答编造 | 影响可信度 | prompt 强制基于 context；无结果时回答不知道；sources 必填 |
| token 泄漏 | 开源项目安全风险 | 提交前扫描 diff；真实值只放本地 `.env` |

---

## 10. 当前下一步

下一步建议执行：

```text
阶段 1：Provider 配置与客户端抽象
```

执行方式：

- 哈宝给 Claude Code 阶段任务。
- Claude Code 写代码和测试。
- 哈宝审查、测试、验证。
- 如果阶段 1 通过，再进入阶段 2。
