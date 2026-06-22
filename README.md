# Deep Research Agent

面向科研人员的学术文献检索与研究报告生成智能体 MVP。

项目方向、质量门和开发优先级以 [GUIDE.md](GUIDE.md) 为准。本 README
主要说明安装、配置和运行方式。

给定一个研究问题，系统自动完成：问题分析 → 检索式生成 → 多源检索 → 去重 → 相关性筛选 → 证据提取 → 缺口分析 → 补充检索 → 中文报告生成 → 引用校验。

## 架构说明

```
用户请求 (POST /api/research)
         │
         ▼
    FastAPI 路由层 (app/api/)
         │
         ▼
    LangGraph 工作流引擎 (app/workflow/)
         │
    ┌────┴──────────────────────────┐
    │                               │
    ▼                               ▼
 服务层 (app/services/)        数据源适配器 (app/providers/)
 - 去重                           - OpenAlex
 - 排序                           - Semantic Scholar
 - 引用校验                        - arXiv
                                  - Crossref
         │                               │
         ▼                               ▼
    LLM 客户端 (app/llm/)          Pydantic 模型 (app/models/)
         │                               │
         ▼                               ▼
    SQLite 持久化 (app/db/)       配置管理 (app/core/)
```

**核心技术栈**：
- **Python 3.11+**：最低 Python 3.11
- **FastAPI**：对外 REST API
- **LangGraph**：研究流程状态机与循环控制
- **Pydantic v2**：结构化数据校验
- **httpx**：异步 HTTP 请求
- **SQLite + aiosqlite**：任务、文献、证据持久化
- **vLLM Qwen3-8B / DeepSeek**：LLM 后端（OpenAI-compatible API，可替换）
- **chat_template_kwargs**：FAST 禁用 / STRONG 启用 Qwen3 reasoning，节省 token

## LangGraph 工作流

```text
initialize
    │
    ▼
plan_queries      ← LLM 生成结构化检索计划
    │
    ▼
search_sources    ← 异步并行检索 4 个学术数据源
    │
    ▼
normalize_and_deduplicate  ← DOI/标题/Provider ID 三级去重
    │
    ▼
rank_and_select   ← 确定性预筛选 + LLM 结构化相关性判断
    │
    ▼
Evidence Engine   ← abstract / FTS5 / PaperQA2 可替换后端
    │
    ▼
extract_evidence  ← passage 级证据与 exact quote 验证
    │
    ▼
assess_gaps       ← 评估证据缺口，决定是否补充检索
    │
    ├── 需要补充 → supplementary_search → rank_and_select (循环)
    │
    └── 不需要 → build_claims → synthesize_report
                             → validate_citations → finalize
```

**停止条件**（防止无限循环）：
- 最多 2 轮补充检索（共 3 轮）
- 新增有效文献过低时停止
- 无证据缺口时直接生成报告

## 快速开始

### 1. 安装依赖

```bash
cd deep-research-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn langgraph pydantic httpx aiosqlite sqlalchemy python-dotenv pytest pytest-asyncio
cp .env.example .env
```

推荐直接安装项目依赖：

```bash
pip install -e ".[dev]"

# 可选：本地 PDF 解析
pip install -e ".[fulltext]"

# 可选：PaperQA2 Evidence Engine
pip install -e ".[paperqa]"
```

### 2. 选择 LLM 后端

**vLLM Qwen3-8B（需 SSH 隧道）**：
```bash
ssh -L 18004:127.0.0.1:8004 sjtu-a800  # 终端 1，保持运行
uvicorn app.main:app --host 127.0.0.1 --port 8000  # 终端 2
```

**Mock 模式（无需 API）**：
```bash
MOCK_MODE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## .env 配置

```env
# vLLM Qwen3-8B
LLM_BASE_URL=http://127.0.0.1:18004/v1
LLM_API_KEY=EMPTY
LLM_MODEL_FAST=qwen3-8b-budget
LLM_MODEL_STRONG=qwen3-8b-budget

# Token limits (8K context; prompt budget = 8192 - max_tokens)
LLM_FAST_MAX_TOKENS=1024
LLM_STRONG_MAX_TOKENS=4096

# Thinking: FAST=off (no reasoning overhead), STRONG=on (better quality)
LLM_FAST_ENABLE_THINKING=false
LLM_STRONG_ENABLE_THINKING=true

# Optional: DeepSeek API
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_API_KEY=sk-your-key
# LLM_MODEL_FAST=deepseek-chat

# OpenAlex / Semantic Scholar (optional)
OPENALEX_API_KEY=
OPENALEX_MAILTO=
SEMANTIC_SCHOLAR_API_KEY=

# Limits
DATABASE_URL=sqlite+aiosqlite:///./storage/app.db
MAX_SEARCH_ROUNDS=3
MAX_PAPERS_PER_SOURCE=20
MAX_SELECTED_PAPERS=20

# Evidence backend: abstract | fts | paperqa
EVIDENCE_BACKEND=abstract
ENABLE_FULL_TEXT=false
PDF_PARSER_BACKEND=pymupdf
HOST=0.0.0.0
PORT=8000
```

## API

启动后访问 `http://127.0.0.1:8000/` 可使用内置研究前端。前端支持任务
提交、阶段进度、论文、Evidence、Claims 和最终报告查看。

### 创建研究任务

```http
POST /api/research
Content-Type: application/json

{
  "topic": "How does dialogue history affect reasoning reliability in large language models?",
  "year_from": 2020,
  "year_to": 2026,
  "max_papers": 12,
  "research_depth": "standard",
  "evidence_backend": "abstract",
  "enable_full_text": false,
  "report_language": "zh-CN"
}
```

核心参数：

| 参数 | 含义 | 推荐默认 |
|---|---|---|
| `topic` / `question` | 研究问题 | 必填 |
| `year_from` | 起始年份 | 留空 |
| `year_to` | 截止年份 | 当前年份 |
| `max_papers` | 最终报告最多纳入论文数 | 12 |
| `research_depth` | `quick` / `standard` / `deep` | standard |

高级参数：

| 参数 | 含义 |
|---|---|
| `evidence_backend` | `abstract`、`fts` 或 `paperqa` |
| `enable_full_text` | 是否下载和解析开放访问全文 |
| `report_language` | `zh-CN` 或 `en` |

`max_papers` 是最终纳入数量，不是候选检索数量。系统会检索更大的候选池，
经过去重、筛选和 evidence quality gate 后再限制最终论文数。

返回：

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "pending"
}
```

### 查询任务状态

```http
GET /api/research/{task_id}
```

返回任务当前阶段、检索轮次、文献数量、warnings、errors。

### 获取研究报告

```http
GET /api/research/{task_id}/report
```

返回 Markdown 格式报告与结构化参考文献。

### 获取文献列表

```http
GET /api/research/{task_id}/papers
GET /api/research/{task_id}/papers?selected_only=true
```

### 获取证据提取

```http
GET /api/research/{task_id}/evidence
GET /api/research/{task_id}/claims
```

### 健康检查

```http
GET /health
```

## 测试

```bash
# Mock 测试（无需 API，52 个用例）
.venv/bin/python -m pytest tests/ -v -k "not test_real"

# vLLM 联调测试（需要先开隧道）
ssh -L 18004:127.0.0.1:8004 sjtu-a800
.venv/bin/python -m pytest tests/test_real_llm.py -v

# 全部测试
.venv/bin/python -m pytest tests/ -v
```

## 基线与评测

固定问题集位于 `evals/questions.json`。其中 `seed` 案例已经固定问题和
覆盖点。`evals/gold_annotations.json` 保存了当前 20 个问题的人工策展
gold paper anchors、DOI 和其证据角色；它们用于 Recall@K，而不是声称
穷尽所有相关文献。

```bash
# 可重复的离线 Mock 基线
.venv/bin/python -m app.evaluation.baseline \
  --mock \
  --backend abstract \
  --output evals/baselines/latest.json

# 评分
.venv/bin/python -m app.evaluation.runner \
  --artifact evals/baselines/latest.json

# PaperQA2 对比（需安装可选依赖并允许全文下载）
.venv/bin/python -m app.evaluation.baseline \
  --backend paperqa \
  --full-text \
  --output evals/baselines/paperqa.json
```

测试覆盖：
1. DOI / 标题规范化 / Provider ID 去重
2. Provider 响应规范化与降级
3. 检索轮次停止条件
4. 引用映射校验与虚构引用检测
5. Mock 模式完整工作流集成测试
6. vLLM /v1/models 验证
7. vLLM 结构化 JSON 输出验证
8. enable_thinking 模式正确性

## 项目结构

```text
deep-research-agent/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI 路由
│   ├── core/
│   │   └── config.py          # 环境变量配置
│   ├── db/
│   │   ├── database.py        # 数据库引擎与会话
│   │   ├── models.py          # SQLAlchemy ORM 模型
│   │   └── repository.py      # CRUD 操作
│   ├── llm/
│   │   ├── client.py          # LLM 客户端抽象
│   │   └── mock_client.py     # Mock LLM（无 Key 时自动使用）
│   ├── models/
│   │   ├── paper.py           # Paper 统一数据模型
│   │   ├── search_plan.py     # 检索计划模型
│   │   ├── evidence.py        # 证据与缺口分析模型
│   │   └── task.py            # 任务状态与指标模型
│   ├── providers/
│   │   ├── base.py            # Provider 抽象基类
│   │   ├── openalex.py        # OpenAlex 适配器
│   │   ├── semantic_scholar.py # Semantic Scholar 适配器
│   │   ├── arxiv.py           # arXiv 适配器
│   │   ├── crossref.py        # Crossref 适配器
│   │   └── mock_provider.py   # Mock Provider（测试/演示）
│   ├── services/
│   │   ├── dedup.py           # 文献去重（DOI/标题/Provider ID）
│   │   ├── ranking.py         # 文献排序与筛选
│   │   └── citation_validation.py # 引用真实性校验
│   ├── workflow/
│   │   └── graph.py           # LangGraph 工作流定义
│   └── main.py                # FastAPI 应用入口
├── tests/
│   ├── conftest.py            # 测试 fixtures
│   ├── test_dedup.py          # 去重测试
│   ├── test_providers.py      # Provider 测试
│   ├── test_citations.py      # 引用校验测试
│   └── test_workflow.py       # 工作流集成测试
├── storage/                   # SQLite 数据库存储目录
├── .env.example               # 环境变量模板
├── pyproject.toml             # 项目元数据与依赖
└── README.md                  # 本文件
```

## 当前限制

1. **8K 上下文约束**：Qwen3-8B-budget max_model_len=8192，报告最多放入 12 篇论文摘要
2. **Reasoning 开销**：STRONG 模式下约 50% token 用于内部推理，延迟和成本较高
3. **后台任务**：服务重启后任务标记为 interrupted，不自动恢复
4. **单数据源模式**：当前仅用 vLLM 单端点，FAST/STRONG 共享同一模型
5. **默认摘要模式**：稳定基线默认使用 `abstract` Evidence Engine
6. **全文后端仍需真实评测**：FTS5 是实验 adapter；PaperQA2 是可选 adapter
7. **Citation correctness 尚不完整**：目前已具备 quote verification 和
   Claim-Evidence 约束，但还没有完整的逐句 entailment verifier

## 下一步扩展

1. **接入真实 API**：配置 `.env` 中的 API Key 即可切换
2. **向量检索**：引入 embedding 模型提升相关性排序
3. **PDF 全文解析**：通过 Open Access URL 获取并解析全文
4. **更多数据源**：PubMed、IEEE Xplore、Tavily 等
5. **多语言支持**：英文检索 + 中文报告已有，可扩展更多语言
6. **任务恢复**：服务重启后自动恢复中断任务
7. **流式输出**：SSE 实时推送研究进度
8. **用户管理**：多用户、认证、任务历史
