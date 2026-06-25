# LangGraph Workflow

## 模式路由

```mermaid
graph TD
    START((START)) --> initialize
    initialize["<b>initialize</b><br/>初始化任务<br/>research_mode 默认=quick"]

    initialize --> route{{"<b>_route_by_mode</b>"}}
    route -->|quick| q1
    route -->|strict| s1

    subgraph QUICK["🔵 QUICK RESEARCH<br/>Tavily Search + Extract<br/>12 节点 · 最多 2 轮补充"]
        q1["<b>classify_question</b><br/>LLM FAST<br/>→ AnswerSchema"]
        q1 --> q2
        q2["<b>quick_plan_queries</b><br/>LLM FAST<br/>→ PlannedQuery[] ≤6"]
        q2 --> q3
        q3["<b>tavily_search</b><br/>Tavily Advanced Search<br/>URL 去重 · 合并 purpose"]
        q3 --> q4
        q4["<b>quick_select_sources</b><br/>评分 · 来源分类<br/>多样性约束 · 10-20 来源"]
        q4 --> q5
        q5["<b>tavily_extract</b><br/>Tavily Extract 正文<br/>失败 → snippet fallback"]
        q5 --> q6
        q6["<b>build_research_notes</b><br/>LLM FAST per-source<br/>quote/数字 程序化校验"]
        q6 --> q7
        q7["<b>quick_assess_coverage</b><br/>LLM STRONG<br/>CoverageAssessment"]
        q7 --> q_decision{{"<b>补充检索?</b><br/>最多 2 轮"}}
        q_decision -->|不足| q_supp
        q_supp["<b>quick_supplementary_search</b>"]
        q_supp --> q3
        q_decision -->|充足| q8
        q8["<b>build_comparison_matrix</b><br/>LLM STRONG<br/>ComparisonRow[]"]
        q8 --> q9
        q9["<b>synthesize_quick_report</b><br/>LLM STRONG · thinking=on<br/>中文报告 · [S#] 引用<br/>max=4096"]
        q9 --> q10
        q10["<b>lightweight_citation_check</b><br/>程序化校验<br/>[S#] 存在性 + 数字可追溯"]
        q10 --> q11
        q11["<b>quick_finalize</b><br/>保存 run_record + report.md"]
        q11 --> END
    end

    subgraph STRICT["🟠 STRICT RESEARCH<br/>学术 PDF + 全文验证<br/>14 节点 · 最多 3 轮补充"]
        s1["<b>plan_queries</b><br/>LLM FAST ×2<br/>SearchPlan + SearchIntent"]
        s1 --> s2
        s2["<b>search_sources</b><br/>OpenAlex · S2<br/>Provider 编译查询"]
        s2 --> s3
        s3["<b>normalize_and_deduplicate</b><br/>DOI → Provider ID → 标题<br/>RRF 多源融合"]
        s3 --> s4
        s4["<b>rank_and_select</b><br/>LLM FAST 终判 → top-15"]
        s4 --> s5
        s5["<b>download_pdfs</b><br/>OA 下载 · SHA-256 缓存"]
        s5 --> s6
        s6["<b>parse_and_chunk</b><br/>Docling/PyMuPDF → FTS5"]
        s6 --> s7
        s7["<b>extract_evidence</b><br/>EvidenceEngine<br/>per-sub-question retrieve"]
        s7 --> s8
        s8["<b>validate_evidence</b><br/>10 项确定性校验"]
        s8 --> s9
        s9["<b>assess_gaps</b><br/>LLM STRONG<br/>thinking=on · max=2048"]
        s9 --> s_decision{{"<b>补充检索?</b><br/>最多 3 轮"}}
        s_decision -->|yes| s_supp
        s_supp["<b>supplementary_search</b>"]
        s_supp --> s3
        s_decision -->|no| s10
        s10["<b>build_claims</b><br/>Claim-Evidence 绑定"]
        s10 --> s11
        s11["<b>build_literature_relations</b><br/>共识/矛盾/互补"]
        s11 --> s12
        s12["<b>synthesize_report</b><br/>LLM STRONG · thinking=on<br/>ValidatedClaims only<br/>max=2048"]
        s12 --> s13
        s13["<b>validate_citations</b><br/>引用编号 + 证据链"]
        s13 --> s14
        s14["<b>finalize</b><br/>保存 run_record + report.md"]
        s14 --> END
    end

    END((END))
```

## 数据流对比

| | Quick | Strict |
|---|---|---|
| **问题分析** | AnswerSchema (LLM) | SearchPlan + SearchIntent (LLM ×2) |
| **内容来源** | Tavily 公开网页 + 论文页面 | OpenAlex + Semantic Scholar API |
| **全文获取** | Tavily Extract (网页正文) | PDF 下载 + Docling 解析 |
| **检索** | — | FTS5 + Dense + RRF + CrossEncoder |
| **证据** | ResearchNote (LLM 结构化) | EvidenceCard → ValidatedClaim |
| **中间产物** | ComparisonRow[] | LiteratureRelations + Claims |
| **循环条件** | CoverageAssessment (维度覆盖) | GapAnalysis (证据缺口) |
| **最大轮次** | 2 (QUICK_MAX_SEARCH_ROUNDS) | 3 (MAX_SEARCH_ROUNDS) |
| **报告依赖** | Notes + Matrix + 来源元数据 | ValidatedClaims only |
| **引用格式** | `[S1]` | `[P1]` |
| **质量检查** | 轻量 [S#] 存在性 + 数字可追溯 | 完整引用完整性 + 证据链 |

## 节点分组

| 颜色 | 阶段 | Quick 节点 | Strict 节点 |
|---|---|---|---|
| 🔵 蓝 | 生命周期 | quick_finalize | initialize · finalize |
| 🟠 橙 | 问题分析 | classify_question · quick_plan_queries | plan_queries |
| 🟢 绿 | 搜索获取 | tavily_search · quick_select_sources · tavily_extract | search_sources · normalize · rank |
| 🟣 紫 | 全文/提取 | build_research_notes | download_pdfs · parse_and_chunk · extract_evidence |
| 🔴 粉 | 评估/缺口 | quick_assess_coverage · quick_supplementary_search | validate_evidence · assess_gaps · supplementary_search |
| 🟦 青 | 报告生成 | build_comparison_matrix · synthesize_quick_report | build_claims · build_literature_relations · synthesize_report |
| 🟡 黄 | 质量门 | lightweight_citation_check | validate_citations |

## LLM 调用

### Quick 模式

| 节点 | 模型 | thinking | max_tokens |
|---|---|---|---|
| classify_question | FAST | off | 1024 |
| quick_plan_queries | FAST | off | 1024 |
| build_research_notes (per-source) | FAST | off | 1024 |
| quick_assess_coverage | STRONG | off | 2048 |
| build_comparison_matrix | STRONG | off | 4096 |
| synthesize_quick_report | STRONG | on | 4096 |

### Strict 模式

| 节点 | 模型 | thinking | max_tokens |
|---|---|---|---|
| plan_queries (×2) | FAST | off | 1024 |
| rank_and_select | FAST | off | 1024 |
| assess_gaps | STRONG | on | 2048 |
| synthesize_report | STRONG | on | 2048 |
