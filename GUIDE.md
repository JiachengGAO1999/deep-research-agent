# Deep Research Agent 总指挥指南

> 本文档是本项目后续产品、架构、开发与评测工作的最高层协作基线。
>
> 适用对象：项目负责人、产品角色、架构角色、开发角色、测试角色，以及参与本仓库工作的 AI Agent。
>
> 当前基线日期：2026-06-22

## 1. 文档权威与使用方式

当不同文档或实现发生冲突时，按以下优先级处理：

1. 用户在当前任务中的明确要求；
2. 本 `GUIDE.md`；
3. 已接受的架构决策记录（ADR）；
4. 测试、评测集与接口契约；
5. `README.md`；
6. 当前代码实现。

代码“已经这样写了”不代表设计已经被接受。当前实现可以被审计、替换或降级为实验分支。

每次开始较大的开发任务前，参与者必须先回答：

1. 本次工作服务于哪个阶段目标？
2. 改动影响哪个质量指标或用户价值？
3. 是否已有成熟库、论文实现或公共服务可以复用？
4. 如何验证改动没有让检索、证据或报告质量退化？
5. 失败时如何回退，是否会破坏现有可运行链路？

如果无法回答，应先补充调研、评测或 ADR，不应直接扩展主链路。

---

## 2. 项目使命

本项目要构建一个面向科研文献的 Deep Research Agent：

```text
研究问题
→ 可解释的问题分解与检索计划
→ 高召回的学术文献发现
→ 文献身份解析、去重与元数据增强
→ 全文级证据检索
→ 带来源位置的结构化证据
→ 跨文献比较、矛盾与缺口分析
→ 逐论断可追溯的研究报告
```

MVP 的目标不是节点数量多，也不是模拟“像人在研究”的过程，而是：

- 找到足够相关且真实存在的文献；
- 正确提取支持问题的原文证据；
- 让报告中的重要论断可追溯到具体文献与页面；
- 在可接受的时间和成本下稳定完成任务；
- 使用可重复评测证明效果不低于选定基线。

项目的核心产品承诺是：

> 每个重要事实都能回答“它来自哪里、原文是什么、系统为何采信它”。

---

## 3. 当前实现的诚实基线

### 3.1 当前阶段

当前项目处于：

> Phase 2 可演示，Phase 3 全文链路原型施工中。

已经具备：

- FastAPI API；
- LangGraph 工作流骨架；
- OpenAlex、Semantic Scholar、arXiv、Crossref 适配器；
- DOI、Provider ID、标题相似度去重；
- 确定性预筛选和 LLM 相关性判断；
- 摘要级结构化证据提取；
- 补充检索循环与停止条件；
- 确定性参考文献生成；
- 引用编号存在性校验；
- SQLite 持久化；
- vLLM/OpenAI-compatible LLM 接口；
- Mock 模式和 Phase 1/2 单元测试。

正在施工、尚不能视为稳定能力：

- PDF 下载与生命周期管理；
- Docling/PyMuPDF 解析；
- DocumentChunk 与 SQLite FTS5；
- 全文 excerpt 注入 evidence extraction；
- evidence provenance 字段；
- Phase 3 测试。

### 3.2 当前质量判断

| 维度 | 当前判断 | 主要原因 |
|---|---|---|
| 工作流骨架 | 可用 | 主链路和循环已经建立 |
| 文献身份与元数据 | 初步可用 | 多源归一化和去重已实现 |
| 文献发现 | 偏弱 | 查询未按 Provider 编译，召回无基准 |
| 文献筛选 | 偏弱 | 关键词启发式和单次 LLM 排序 |
| 全文检索 | 原型 | 当前主要是 FTS5 关键词 OR |
| 证据溯源 | 未闭环 | provenance 字段未真正绑定和验证 |
| 报告综合 | 摘要级 | 最终生成仅使用少量截断摘要与 findings |
| 引用可靠性 | 部分可用 | 只验证编号存在，不验证论断支持关系 |
| 效果评测 | 缺失 | 没有固定问题集、gold evidence 或基线对比 |
| 服务可靠性 | 开发级 | 进程内后台任务，无恢复与持久化 checkpoint |

### 3.3 不得误用的能力名称

在相应质量门完成前：

- `citation_validation` 只能解释为“引用编号和参考文献完整性校验”，不能称为“事实正确性校验”；
- LLM 输出的 `evidence_quote` 未经原文匹配前，不能标记为 `direct_quote`；
- FTS5 返回 chunk 不等于“语义相关证据”；
- PDF 成功下载或解析不等于“全文能力完成”；
- Mock workflow 通过不等于真实文献研究效果达标；
- 不得对外宣称“不输 Deep Research”或“达到专家水平”。

---

## 4. 核心工程原则

### 4.1 证据优先于生成

系统首先生产可验证的证据对象，然后才生成报告。

禁止让最终写作模型直接从大量摘要或原始 chunk 自由发挥，再用引用编号包装结果。

### 4.2 Claim-Evidence 是系统主干

最终系统的核心数据流应是：

```text
SubQuestion
  → RetrievedPassage[]
  → Evidence[]
  → Claim[]
  → ReportSection[]
```

而不是：

```text
SearchResult[]
  → 一个长 Prompt
  → Report
```

### 4.3 编排层与能力层解耦

本项目优先保留和发展：

- 产品 API；
- LangGraph 编排；
- 任务状态；
- 多轮研究策略；
- Provider 调度；
- 证据、论断和报告的数据契约；
- 可观测性与评测。

全文解析、索引、retrieval、reranking、metadata enrichment 等成熟能力优先通过适配器复用，不与工作流代码耦合。

### 4.4 质量必须可测量

任何“效果更好”的结论必须来自固定数据集上的前后对比。

人工看一两份报告、Mock 测试通过、Prompt 看起来合理，均不能作为效果证明。

### 4.5 保持可退化，但不可静默降质

Provider、PDF 或 LLM 失败时可以降级，但必须：

- 在任务结果中明确记录降级；
- 标记证据来源是 abstract 还是 full text；
- 不把 fallback 结果伪装成高置信度结果；
- 统计降级比例；
- 允许质量门据此阻止任务标记为高质量完成。

---

## 5. 自研与复用边界

### 5.1 优先复用

以下模块已有成熟开发范式，除非有评测和 ADR 支持，否则禁止从零重写生产级版本：

| 能力 | 首选方向 | 本项目职责 |
|---|---|---|
| 科学文献全文 RAG | PaperQA2 或等价成熟实现 | 提供统一 EvidenceEngine 适配器 |
| PDF 结构化解析 | Docling/成熟 reader | 配置、封装、缓存和质量检测 |
| OA 地址和许可解析 | Unpaywall/权威元数据源 | 调度与字段归一化 |
| 通用模型适配 | LiteLLM 或稳定 OpenAI-compatible adapter | 模型角色配置和成本策略 |
| 文献身份与引用格式 | Crossref/S2/OpenAlex、成熟 bibliography 工具 | 统一 Paper 模型 |
| Hybrid retrieval | 成熟 BM25/vector/reranker 组件 | 评测和业务参数配置 |
| 工作流持久化 | LangGraph checkpointer 或成熟任务队列 | 任务生命周期和 API |

### 5.2 允许自研

以下部分体现本项目差异化，可持续自研：

- 面向用户问题的研究任务建模；
- 学术 Provider 的调度策略和查询编译；
- 搜索轮次、停止条件与预算控制；
- 子问题覆盖与 gap analysis；
- Evidence、Claim、ReportSection 数据契约；
- 跨论文一致性、矛盾和证据强度分析；
- 中文科研报告的结构与交互体验；
- 产品级任务状态、进度、审计记录和可观测性；
- 适合本项目目标的评测集与质量门。

### 5.3 引入依赖的决策要求

引入或拒绝一个成熟组件时，至少记录：

- 组件版本和许可证；
- 它替代了哪些自研代码；
- 输入输出契约；
- 性能、成本和部署要求；
- 最小对比实验；
- 回退方案；
- 是否需要锁定版本。

重大决定写入 `docs/adr/`。目录不存在时可以在首次 ADR 中创建。

---

## 6. 目标架构

```text
API / Channel
    │
    ▼
Task Orchestrator
    │
    ├── Research Planner
    │      └── Question → SubQuestions + SearchIntents
    │
    ├── Literature Discovery
    │      ├── Provider-specific Query Compilers
    │      ├── OpenAlex / S2 / arXiv / Crossref
    │      └── Identity Resolution + Metadata Enrichment
    │
    ├── Evidence Engine Adapter
    │      ├── OA Resolution / Acquisition
    │      ├── Parse / Index / Cache
    │      ├── Hybrid Retrieval
    │      ├── Reranking
    │      └── Contextual Evidence Extraction
    │
    ├── Evidence & Claim Layer
    │      ├── Exact Quote Verification
    │      ├── Claim-Evidence Binding
    │      ├── Contradiction / Agreement
    │      └── Coverage / Gap Analysis
    │
    ├── Report Composer
    │      ├── Outline
    │      ├── Section Drafting
    │      ├── Cross-section Revision
    │      └── Deterministic References
    │
    └── Quality Gate
           ├── Citation Integrity
           ├── Citation Correctness
           ├── Coverage
           ├── Unsupported Claim Detection
           └── Completion Decision
```

### 6.1 推荐的接口边界

工作流不得直接依赖某个具体全文库的数据类型。应通过稳定接口调用：

```python
class EvidenceEngine:
    async def ingest(self, papers: list[Paper]) -> IngestionResult: ...

    async def retrieve(
        self,
        question: str,
        sub_question: str,
        paper_ids: list[str] | None,
        limit: int,
    ) -> list[RetrievedPassage]: ...

    async def extract(
        self,
        sub_question: str,
        passages: list[RetrievedPassage],
    ) -> list[Evidence]: ...
```

PaperQA2、现有 FTS5 或未来其他实现都应位于该接口后方。

---

## 7. 核心数据契约

### 7.1 Paper

Paper 表示可被唯一识别的文献实体，而不是某个 Provider 的一次搜索结果。

必须逐步支持：

- canonical paper ID；
- DOI、arXiv ID、OpenAlex ID、S2 ID 等外部标识；
- title、authors、year、venue；
- abstract；
- landing page 与 full-text URL 分离；
- OA 状态、许可和获取来源；
- retract/correction 状态；
- 元数据字段来源和冲突记录。

### 7.2 RetrievedPassage

最少字段：

```text
passage_id
paper_id
chunk_id
text
section_title
page_start
page_end
source_url
retrieval_method
retrieval_score
rerank_score
parser_name
document_hash
```

### 7.3 Evidence

Evidence 必须表示“能够支持或反驳某个子问题/论断的来源片段”。

最少字段：

```text
evidence_id
paper_id
passage_id
sub_question_id
exact_quote
normalized_quote
interpretation
stance: supports | contradicts | contextual | inconclusive
evidence_type: direct_quote | close_paraphrase | inference
verification_status
page_start / page_end
confidence
```

硬规则：

- `direct_quote` 必须通过程序化原文匹配；
- 无 passage ID 的内容不能成为 full-text evidence；
- inferred 内容必须引用其推理依赖的多个 Evidence；
- abstract evidence 必须显式标记，不得伪装成 full-text evidence。

### 7.4 Claim

最终报告中的重要事实先成为 Claim：

```text
claim_id
claim_text
claim_type
importance
evidence_ids[]
support_status
confidence
section_id
```

没有证据的事实型 Claim 原则上不得进入最终报告。若确需保留，必须标记为假设、建议或系统推断。

---

## 8. 标准研究工作流

### 8.1 问题分析

输出必须包含：

- 规范化研究问题；
- 范围和时间约束；
- 3–8 个可检索子问题；
- 核心概念、同义词、实体和排除项；
- 预期回答类型；
- 初始搜索预算。

### 8.2 查询编译

禁止把同一 Boolean-like 字符串原样发送到所有 Provider。

必须先生成 Provider 无关的 `SearchIntent`，再分别编译：

```text
SearchIntent
  ├── OpenAlexQuery
  ├── SemanticScholarQuery
  ├── ArxivQuery
  └── CrossrefQuery
```

### 8.3 文献发现

搜索策略至少考虑：

- broad query 与 narrow query；
- 术语变体；
- 时间过滤；
- seminal 与 recent work 的平衡；
- 引用量只能作为弱质量信号；
- 去重后的新增率；
- 子问题覆盖情况。

### 8.4 文献筛选

筛选分为：

1. 确定性硬条件；
2. 低成本候选排序；
3. query-aware reranking；
4. 多样性和子问题覆盖约束。

不允许仅按 citation count 或是否有 abstract 作为 fallback 排序并视为高质量结果。

### 8.5 证据检索与提取

推荐执行方式：

```text
for each sub_question:
    retrieve passages
    rerank passages
    map extract evidence per paper/passage
    verify exact quotes
    aggregate evidence
```

禁止将十几篇论文一次塞入 1024 token 的结构化输出调用并期待完整证据。

### 8.6 Gap Analysis

Gap 分析必须基于：

- 子问题覆盖率；
- 有效 Evidence 数量；
- 独立文献数；
- 是否只有 abstract evidence；
- 是否存在相互矛盾证据；
- 文献时间和方法多样性。

不能只让 LLM 依据自己刚生成的 findings 自我判断“是否搜索充分”。

### 8.7 报告生成

报告采用分阶段生成：

1. 基于 Claim 建立 outline；
2. 按 section 生成草稿；
3. 检查段落中的 Claim-Evidence 绑定；
4. 跨 section 去重与一致性修订；
5. 确定性生成参考文献；
6. 执行最终质量门。

最终写作 Prompt 应传入经过验证的 Evidence/Claim，而不是只传入截断摘要。

---

## 9. 质量门

### 9.1 单条 Evidence 质量门

一条 full-text Evidence 合格需要：

- paper identity 有效；
- passage 可定位；
- exact quote 在来源文本中匹配；
- page 或 section 至少一个可用；
- 与目标子问题相关；
- evidence type 和来源层级正确。

### 9.2 单条 Claim 质量门

重要事实型 Claim 合格需要：

- 至少一个有效 Evidence；
- 引用的 Evidence 语义上支持 Claim；
- 定量结论包含对应原始数字或明确的计算来源；
- 多文献概括没有把单篇结论表述为领域共识；
- 矛盾证据不能被静默忽略。

### 9.3 报告级质量门

至少统计：

- citation precision；
- citation completeness；
- quote verification rate；
- unsupported claim rate；
- sub-question coverage；
- unique supporting papers；
- abstract-only evidence ratio；
- full-text acquisition rate；
- retrieval diversity；
- 任务降级事件；
- 总耗时、LLM tokens 和外部请求成本。

高质量完成建议满足：

```text
orphan citation rate = 0
verified direct quote rate = 100%
unsupported important claim rate = 0
citation completeness ≥ 95%
每个核心子问题至少有 2 篇独立文献支持，或明确声明证据不足
```

阈值可以通过 ADR 调整，但不得在没有记录的情况下静默降低。

---

## 10. 评测体系

### 10.1 基线

至少保留以下对照：

- 当前稳定版本；
- Dify Deep Research 类通用模板；
- PaperQA2 或选定的成熟学术 RAG；
- 直接使用强模型 + 搜索结果的朴素基线。

不能只比较最终文本“读起来哪个好”。必须拆分评测：

1. 文献发现；
2. passage retrieval；
3. evidence extraction；
4. claim attribution；
5. 最终报告。

### 10.2 最小评测集

MVP 阶段建立 20–30 个固定问题，覆盖：

- 单一事实问题；
- 多论文综合问题；
- 方法比较；
- 时间演进；
- 存在争议或矛盾的问题；
- 近期文献问题；
- 低资源或长尾主题；
- 中英文提问。

每个问题至少保存：

- gold/relevant papers；
- 可接受的支持 passage；
- 核心 answer points；
- 必须覆盖的子问题；
- 已知错误和诱导项。

### 10.3 检索指标

至少使用：

- paper Recall@K；
- passage Recall@K；
- MRR 或 nDCG；
- 去重 precision；
- 子问题覆盖率；
- 新增文献收益曲线。

### 10.4 生成指标

至少使用：

- claim correctness；
- citation precision；
- citation completeness；
- evidence entailment；
- unsupported claim rate；
- answer completeness；
- 专家或双盲 pairwise preference。

LLM-as-judge 只能作为辅助，必须抽样人工复核，并固定 judge prompt 和模型版本。

### 10.5 合并门槛

影响检索、证据或报告行为的改动，在进入稳定链路前必须提供：

- 评测命令；
- 基线结果；
- 新结果；
- 成本和延迟变化；
- 失败案例；
- 是否达到预先定义的通过条件。

---

## 11. 开发路线

### Phase A：冻结并测量当前基线

目标：得到一个可重复运行、可比较的摘要级系统。

完成条件：

- Phase 1/2 测试稳定通过；
- Phase 3 不再影响 Mock/摘要级 workflow；
- 修复测试阻塞和 fixture；
- README 与真实实现一致；
- 建立首批固定研究问题；
- 保存当前系统的检索和报告结果。

### Phase B：建立 Evidence Engine 边界

目标：让工作流不直接绑定 FTS5 或具体全文库。

完成条件：

- 定义 `EvidenceEngine`、`RetrievedPassage`、`Evidence`；
- 现有 FTS5 作为实验 adapter；
- 接入 PaperQA2 或经评测选定的成熟实现；
- 完成同一问题上的 adapter 对比；
- 选择默认 evidence backend。

### Phase C：全文证据闭环

目标：实现可验证的 passage → evidence。

完成条件：

- OA resolution；
- PDF/全文获取状态可观测；
- parser 输出带页面；
- per-sub-question retrieval；
- exact quote verification；
- provenance 完整持久化；
- abstract/full-text evidence 明确区分。

### Phase D：Claim-Evidence 报告

目标：让报告中的重要论断可审计。

完成条件：

- Claim 模型；
- Claim-Evidence 绑定；
- 分章节生成；
- citation correctness 检查；
- unsupported claim 拦截；
- 参考文献确定性生成。

### Phase E：检索与研究策略增强

目标：提升召回、覆盖和多轮研究收益。

完成条件：

- Provider-specific query compiler；
- hybrid retrieval 和 reranking；
- 子问题覆盖驱动 supplementary search；
- 搜索预算与收益停止条件；
- 矛盾检测和证据强度表达。

### Phase F：服务可靠性

目标：从开发 demo 进入可持续运行的 MVP。

完成条件：

- 持久化 checkpoint 或任务队列；
- 服务重启恢复；
- 幂等写入；
- 超时与取消；
- SSE/事件流进度；
- Provider/LLM 限流；
- 任务级成本和耗时预算；
- 可复现部署依赖。

---

## 12. 当前优先级

### P0：检索质量（2026-06-23 更新）

当前 `Gold Recall@K ≈ 0`，瓶颈在论文发现阶段。
必须建立逐阶段 Recall 诊断体系，定位 Gold 丢失的具体步骤。

1. **逐阶段 Recall 追踪** — 记录 Provider 原始结果 → 去重 → 过滤 → Rerank → LLM 筛选每一步的 Gold 保留率；
2. **校验 Gold 身份** — 确认 Gold 的 DOI、arXiv ID、规范化标题能在 API 中命中；
3. **Provider-specific Query Compiler** — SearchIntent → OpenAlex filters / arXiv field search / S2 semantic query；
4. **Semantic Scholar 全局限速** — 1 req/s semaphore，Retry-After 响应；
5. **RRF 多源融合** — 跨查询、跨 Provider 的 Reciprocal Rank Fusion + query coverage 信号；
6. **候选池分层控制** — 每源每查询 10-20（非 3），融合去重后 100，Cross-Encoder top 40，LLM top 15。

### P1：证据与报告质量

1. ~~exact quote verification~~ ✅ evidence_validator.py 已实现；
2. ~~Claim-Evidence 模型~~ ✅ ResearchClaim + claim_verifier.py 已实现；
3. **EvidenceCard 结构化字段填充** — paper_role, subject, direction 当前为 unknown，需 LLM 从 chunk 提取；
4. **SentenceAudit 接入** — 模型已定义，未接入 report 后处理；
5. **OA host 白名单扩展** — S2 PDF 仍被拦截，6/10 论文无法获取全文；
6. 20 题完整 baseline 对比（Abstract / FTS / 未来向量后端）。

### P2：产品化

1. 任务恢复；
2. 流式进度；
3. 成本预算；
4. 缓存与并发；
5. 多用户和权限。

### 暂不优先

- 继续美化前端；
- 在没有评测前反复调 Prompt；
- 扩展大量新 Provider；
- 自研向量数据库或 PDF parser；
- 用更多模型掩盖检索 pipeline 的缺陷。

---

## 13. 测试要求

### 13.1 测试层级

```text
Unit
  数据规范化、query compiler、quote verifier、引用解析

Contract
  Provider、LLM、EvidenceEngine adapter

Integration
  搜索→去重→证据→Claim→报告

Evaluation
  固定真实问题上的效果指标

Live Smoke
  真实 Provider、真实 LLM、真实 PDF
```

### 13.2 测试纪律

- 测试不得以“可能搜不到也没关系”结束；
- 成功解析必须断言有预期文本或 chunk；
- 成功检索必须断言目标 passage 排名；
- Mock 数据不得伪装成真实论文；
- Live 测试必须明确标记，默认离线测试不得访问网络；
- 外部依赖应通过 fixture 或 adapter mock；
- 测试数据库和 PDF 缓存不得污染开发数据；
- 所有 fallback 都应有独立测试和显式状态。

---

## 14. 可观测性要求

每个任务应能回答：

- 生成了哪些子问题和查询？
- 每个 Provider 请求了什么、返回多少、失败多少？
- 去重前后分别有多少论文？
- 为什么选择或排除一篇论文？
- 哪些论文获得了全文？
- 每个 parser 的成功率和耗时？
- 每个子问题检索了哪些 passage？
- 哪些 Evidence 通过或未通过验证？
- 每个 Claim 绑定了哪些 Evidence？
- 为什么触发或停止补充检索？
- 最终任务发生过哪些降级？
- 消耗了多少时间、token 和外部请求？

日志不能是唯一载体；关键事件应进入结构化任务记录。

---

## 15. 协作规则

### 15.1 开发任务模板

建议每个较大任务先写清：

```text
目标：
非目标：
当前基线：
复用方案：
设计与接口：
验收指标：
测试计划：
回退方案：
涉及 ADR：
```

### 15.2 代码审查重点

审查顺序：

1. 是否提高或保护用户可见质量；
2. 是否破坏 provenance；
3. 是否重复实现成熟能力；
4. 是否有可测量的验收标准；
5. 是否引入静默降级；
6. 是否保持接口边界；
7. 是否有测试和依赖声明；
8. 最后才是局部代码风格。

### 15.3 文档同步

出现以下变化时必须更新本指南或 ADR：

- 默认 Evidence Engine 变化；
- 核心工作流变化；
- Evidence/Claim 数据契约变化；
- 质量门阈值变化；
- 基线系统变化；
- 阶段完成状态变化；
- 放弃或替换重要组件。

README 负责安装、配置、运行和 API 使用，不承担架构决策记录。

---

## 16. Definition of Done

一个功能只有同时满足以下条件才算完成：

- 代码实现完成；
- 依赖可复现；
- 单元/契约/集成测试按风险补齐；
- 指标或行为验收通过；
- 失败和降级路径已验证；
- provenance 未被破坏；
- 文档与实际行为一致；
- 没有把实验能力表述为稳定能力；
- 对主链路有影响时完成基线对比。

“能运行一次”不等于完成，“LLM 返回了看起来合理的文本”也不等于完成。

---

## 17. MVP 完成定义

项目达到“可信学术 Deep Research MVP”至少需要：

- 20–30 个固定真实问题的评测集；
- 多源文献发现和 Provider-specific query；
- 稳定的文献身份解析与去重；
- 成熟全文 Evidence Engine；
- passage 级检索与 reranking；
- exact quote 和页面 provenance；
- Claim-Evidence 报告生成；
- citation precision/completeness 质量门；
- 与 Dify 类模板及 PaperQA2 的可重复对比；
- 可恢复或至少可明确中断的任务执行；
- 完整的成本、耗时和降级记录。

只有在固定评测集上达到预设门槛后，才可以使用“不输某基线”的表述。

---

## 18. 当前总指挥决策摘要

截至 2026-06-23，项目采用以下决策：

### 架构与编排
1. FastAPI + LangGraph 15 节点工作流 + 多源 Provider + 统一 Paper 模型；
2. Evidence Engine adapter（abstract/FTS/paperqa），核心工作流不绑定具体库；
3. 20 题评测集 + 58 篇 gold papers 作为检索 ground truth；

### 证据管线（本轮完成）
4. EvidenceCard 结构化字段：subject, metric, value, direction, comparison, scope, paper_role, is_inference；
5. 确定性 evidence_validator：10 项检查（quote 逐字匹配、数字一致性、方向合理性、枚举合法性）；
6. Claim 两阶段验证：确定性 + 独立 LLM 蕴含判断（不同 prompt，不自我批准）；
7. `validate_evidence` → `build_claims` → `build_literature_relations` 三节点管线；
8. 报告只读 validated Claim，不读 raw chunks 或未验证 EvidenceCard；
9. System prompt 改为 XML 结构化策略（role/source_of_truth/claim_policy/paper_role_policy/meta_claim_policy/final_check）；

### 检索
10. FTS5：OR 语义 + per-paper 检索 + 停用词过滤 + 短语引用（query_builder.py）；
11. 补充检索路径：`supplementary → normalize_and_deduplicate → rank`；
12. 每源每查询 10-20，S2 限速 1 req/s；

### 默认配置
13. PDF 解析默认 Docling（PyMuPDF fallback），证据默认 abstract；
14. Run record 自动保存 `storage/tasks/{task_id}/`（run_record.json + report.md）；

### 已知限制
15. OA host 白名单过严 — S2 PDF 被拦截；
16. EvidenceCard 结构化字段（paper_role 等）当前为 unknown — 需 LLM 提取填充；
17. SentenceAudit 模型已定义未接入工作流；
18. STRONG LLM 调用受 8K 上下文限制（报告 max_tokens=2048，缺口分析=2048）；
19. Gold Recall@K ≈ 0 — 检索词汇匹配问题待解决。

这份指南不是静态愿景文档。项目每完成一个阶段，应更新当前基线、决策摘要和下一优先级。
