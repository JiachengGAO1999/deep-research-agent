# Deep Research Agent

面向科研人员的学术文献检索与研究报告生成智能体。支持两种研究模式：

- **Quick Research**（默认）：基于 Tavily 搜索公开网页、论文页面和摘要，生成带真实来源引用的完整研究报告
- **Strict Research**：下载 PDF 全文、解析、FTS5 检索、EvidenceCard 验证，生成逐句可追溯的严格报告

项目方向和开发决策以 [GUIDE.md](GUIDE.md) 为准。工作流图见 [docs/workflow.md](docs/workflow.md)。

## 快速开始（新用户）

### 1. 前置条件

- Python 3.11+
- Quick 模式：Tavily API key（[免费注册](https://tavily.com)）+ 任意 OpenAI-compatible LLM
- Strict 模式（可选）：OpenAlex / Semantic Scholar API key

### 2. 克隆并安装

```bash
git clone git@github.com:JiachengGAO1999/deep-research-agent.git
cd deep-research-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，至少填两项：
#   LLM_API_KEY=sk-your-key        # 你的 LLM API key
#   TAVILY_API_KEY=tvly-your-key   # 你的 Tavily API key
```

支持任意 OpenAI-compatible LLM（DeepSeek、OpenAI、vLLM 等），只需改 `LLM_BASE_URL` 和 `LLM_MODEL_*`。

### 4. 运行

```bash
# 启动服务器
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或 Mock 模式（无需任何 API key，使用假数据验证流程）
MOCK_MODE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://localhost:8000`，输入研究问题，选择研究模式，提交即可。

### 5. 运行测试

```bash
# Mock 测试（无需 API，85 个用例）
.venv/bin/python -m pytest tests/ -v -k "not test_real"
```

## 研究模式

| | Quick | Strict |
|---|---|---|
| **内容来源** | Tavily 公开网页 + 论文页面 | OpenAlex + Semantic Scholar API |
| **全文** | Tavily Extract（网页正文） | PDF 下载 + Docling/PyMuPDF 解析 |
| **证据** | ResearchNote（LLM 结构化提取） | EvidenceCard → ValidatedClaim（10 项校验） |
| **引用** | `[S1]` 格式，指向真实 URL | `[P1]` 格式，指向论文 DOI |
| **报告** | 中文综合报告，可读性强 | 严格逐句可追溯，覆盖率取决于全文获取率 |
| **适用** | 快速了解领域概貌 | 需要逐句可审计的严格证据 |

默认使用 Quick 模式。在 Web 前端「高级设置」或 API 参数中可切换。

## API

### 创建研究任务

```http
POST /api/research
Content-Type: application/json

{
  "research_question": "Which RAG techniques most consistently reduce factual hallucination?",
  "research_mode": "quick",
  "year_from": 2022,
  "year_to": 2026,
  "max_papers": 12,
  "report_language": "zh-CN"
}
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `research_question` | 研究问题 | 必填 |
| `research_mode` | `quick` / `strict` | `quick` |
| `year_from` / `year_to` | 年份范围 | 不限 |
| `max_papers` | 最终纳入论文数（Strict 模式） | 12 |
| `report_language` | `zh-CN` / `en` | `zh-CN` |

### 查询状态和结果

```http
GET /api/research/{task_id}          # 任务状态 + 进度
GET /api/research/{task_id}/report   # 研究报告
GET /api/research/{task_id}/papers   # 文献列表
GET /api/research/{task_id}/evidence # 证据提取
GET /api/research/{task_id}/claims   # Claim-Evidence
```

## 项目结构

```text
deep-research-agent/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── api/routes.py              # REST 接口
│   ├── core/config.py             # 环境变量配置
│   ├── models/
│   │   ├── task.py                # TaskState, TaskMetrics
│   │   ├── paper.py               # Paper 统一模型
│   │   ├── evidence.py            # EvidenceCard, Claim, GapAnalysis
│   │   ├── search_plan.py         # SearchPlan (Strict)
│   │   ├── search_intent.py       # SearchIntent (Strict)
│   │   └── quick_research.py      # Quick 模式模型 (AnswerSchema, ResearchNote...)
│   ├── workflow/
│   │   ├── graph.py               # LangGraph 主图 (28 节点, 双模式路由)
│   │   └── quick_research.py      # Quick Research 子图 (12 节点)
│   ├── clients/__init__.py        # Tavily Search + Extract 客户端
│   ├── providers/                 # 学术 API 适配器 (OpenAlex, S2, arXiv)
│   ├── services/                  # 去重, 排序, 证据引擎, PDF, FTS5, 引用校验
│   ├── llm/
│   │   ├── client.py              # OpenAI-compatible LLM 客户端
│   │   └── mock_client.py         # Mock LLM (MOCK_MODE=1)
│   ├── db/                        # SQLite 持久化
│   └── static/                    # Web 前端
├── tests/
│   ├── test_quick_research.py     # Quick 模式测试 (27)
│   ├── test_workflow.py           # Strict 模式测试 (9)
│   └── ...                        # 去重, 引用, Provider 等测试
├── docs/workflow.md               # 工作流 Mermaid 图
├── .env.example                   # 环境变量模板
└── pyproject.toml
```

## 当前限制

1. **Quick 模式**：报告质量依赖 Tavily 搜索覆盖率和 LLM 综合能力，结论未经 PDF 全文逐句核验
2. **Strict 模式**：PDF 获取成功率受 OA 限制，全文覆盖率可能较低
3. **无状态执行**：服务重启后任务不恢复
4. **单用户**：无认证和权限管理
5. **Strict 全文后端需真实评测**：FTS5 是实验性 adapter

## 测试覆盖

```bash
# Mock 测试（无需任何外部服务，85 个用例全部通过）
.venv/bin/python -m pytest tests/ -v -k "not test_real"

# 包含: 去重(14) · Provider(6) · 引用校验(12) · API参数(4) ·
#       Strict工作流(9) · Quick工作流(27) · 查询编译(2) · 其他(11)
```
