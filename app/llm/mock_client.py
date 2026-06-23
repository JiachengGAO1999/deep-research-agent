"""Mock LLM client that returns realistic structured responses without API calls."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Pre-built mock responses for each workflow stage
MOCK_SEARCH_PLAN_JSON = {
    "research_topic": "Impact of dialogue history on reasoning reliability in large language models",
    "core_concepts": [
        "dialogue history",
        "multi-turn reasoning",
        "LLM reasoning reliability",
        "context window effects",
        "reasoning degradation",
    ],
    "synonyms": {
        "dialogue history": ["conversation history", "chat history", "multi-turn dialogue", "context accumulation"],
        "reasoning reliability": ["reasoning accuracy", "logical consistency", "reasoning quality", "inference stability"],
        "large language models": ["LLMs", "GPT", "transformer models", "language models"],
    },
    "queries": [
        {
            "query_string": "dialogue history multi-turn reasoning reliability large language models",
            "rationale": "Direct search for the core research question",
            "keywords": ["dialogue history", "multi-turn reasoning", "reliability", "LLM"],
        },
        {
            "query_string": "context accumulation reasoning degradation LLM conversation",
            "rationale": "Focus on the degradation aspect of multi-turn reasoning",
            "keywords": ["context accumulation", "reasoning degradation", "conversation"],
        },
        {
            "query_string": "multi-turn logical consistency language model evaluation",
            "rationale": "Search for evaluation methods and metrics",
            "keywords": ["logical consistency", "multi-turn", "evaluation"],
        },
        {
            "query_string": "chat history effect inference quality transformer models",
            "rationale": "Broader search including earlier transformer architectures",
            "keywords": ["chat history", "inference quality", "transformer"],
        },
        {
            "query_string": "reasoning faithfulness context window long conversation LLM",
            "rationale": "Target the faithfulness and context window angle",
            "keywords": ["reasoning faithfulness", "context window", "long conversation"],
        },
    ],
    "year_from": 2020,
    "year_to": 2026,
    "domains": ["cs.CL", "cs.AI"],
    "criteria": {
        "include": [
            "Studies on multi-turn reasoning in LLMs",
            "Papers evaluating reasoning reliability across conversation turns",
            "Work on context window and reasoning quality",
            "Empirical evaluations with quantitative metrics",
        ],
        "exclude": [
            "Single-turn reasoning only",
            "Non-English language models without broader applicability",
            "Pure engineering/implementation papers without evaluation",
        ],
    },
}

MOCK_RANKING_RESULTS = [
    {"internal_id": "mock_p1", "relevance_score": 95, "include": True, "reason": "Directly addresses multi-turn reasoning reliability with empirical evaluation", "matched_aspects": ["multi-turn reasoning", "reliability", "empirical evaluation"]},
    {"internal_id": "mock_p2", "relevance_score": 88, "include": True, "reason": "Studies context accumulation effects on reasoning quality", "matched_aspects": ["context accumulation", "reasoning quality"]},
    {"internal_id": "mock_p3", "relevance_score": 85, "include": True, "reason": "Proposes evaluation framework for multi-turn logical consistency", "matched_aspects": ["logical consistency", "evaluation", "multi-turn"]},
    {"internal_id": "mock_p4", "relevance_score": 80, "include": True, "reason": "Analyzes reasoning degradation patterns in long conversations", "matched_aspects": ["reasoning degradation", "long conversations"]},
    {"internal_id": "mock_p5", "relevance_score": 75, "include": True, "reason": "Empirical study on dialogue history impact", "matched_aspects": ["dialogue history", "empirical"]},
    {"internal_id": "mock_p6", "relevance_score": 70, "include": True, "reason": "Investigates attention patterns across conversation turns", "matched_aspects": ["attention", "conversation turns"]},
    {"internal_id": "mock_p7", "relevance_score": 65, "include": True, "reason": "Benchmark for multi-turn reasoning evaluation", "matched_aspects": ["benchmark", "multi-turn reasoning"]},
    {"internal_id": "mock_p8", "relevance_score": 60, "include": True, "reason": "Related work on context window optimization", "matched_aspects": ["context window"]},
    {"internal_id": "mock_p9", "relevance_score": 40, "include": False, "reason": "Focuses on single-turn reasoning only", "matched_aspects": ["single-turn"]},
    {"internal_id": "mock_p10", "relevance_score": 30, "include": False, "reason": "Engineering implementation without evaluation", "matched_aspects": ["engineering"]},
]

MOCK_GAP_ANALYSIS_JSON = {
    "covered_aspects": [
        "General multi-turn reasoning reliability",
        "Context accumulation effects",
        "Evaluation frameworks for dialogue-based reasoning",
    ],
    "gaps": [
        {
            "sub_question": "How does dialogue history length specifically affect reasoning about quantitative problems?",
            "current_coverage": "Existing papers focus on qualitative reasoning tasks",
            "what_is_missing": "Studies with quantitative/mathematical reasoning in multi-turn settings",
            "severity": "medium",
        },
        {
            "sub_question": "What mitigation strategies exist for reasoning degradation in long conversations?",
            "current_coverage": "Papers identify the problem but few propose solutions",
            "what_is_missing": "Intervention studies or architectural modifications to address degradation",
            "severity": "high",
        },
    ],
    "needs_supplementary_search": True,
    "supplementary_queries": [
        "reasoning degradation mitigation strategy long context LLM",
        "multi-turn mathematical reasoning reliability language model",
    ],
    "rationale": "Core aspects are covered but mitigation strategies and quantitative reasoning domains need more evidence.",
}

MOCK_REPORT = """# 研究报告: 对话历史如何影响大语言模型的推理可靠性

## 研究问题

本报告旨在系统性地探讨：在多轮对话场景下，对话历史（dialogue history）的累积如何影响大语言模型（LLM）的推理可靠性和一致性。

## 检索范围与方法

- **数据源**: OpenAlex, Semantic Scholar, arXiv, Crossref
- **检索时期**: 2020–2026
- **检索轮次**: 2 轮
- **检索式数量**: 7 个（初始 5 个 + 补充 2 个）
- **最终纳入文献**: 8 篇

## 核心发现

### 1. 多轮推理存在系统性退化

多项研究表明，随着对话轮次的增加，LLM 的推理准确性呈现系统性下降趋势 [P1][P4]。这种退化主要表现在：
- 逻辑一致性的降低
- 对早期信息的遗忘
- 推理步骤的跳跃和省略

### 2. 上下文累积是主要机制

Wang et al. (2023) 的研究指出，对话历史中的信息累积会导致注意力分散，模型难以有效区分相关信息和干扰信息 [P2]。这种"上下文过载"效应是推理退化的核心机制。

### 3. 评估框架的发展

近期工作提出了多种评估多轮推理可靠性的框架 [P3][P7]。这些框架通常采用：
- 逐步推理追踪
- 跨轮次一致性检查
- 对抗性对话测试

### 4. 注意力的动态变化

研究表明，随着对话的进行，模型的注意力分布发生显著变化 [P6]。早期轮次的注意力更加集中，后期则趋于分散。

## 不同研究之间的一致与分歧

**一致点**:
- 多轮推理确实存在可靠性下降的问题
- 上下文长度是重要影响因素
- 需要专门的评估方法

**分歧点**:
- 退化开始的轮次阈值（3轮 vs 5轮 vs 更多）
- 模型规模是否是保护因素
- 不同推理类型的受影响程度

## 研究局限与证据缺口

1. **定量推理研究不足**: 现有研究主要集中在定性推理任务上，缺乏对数学/定量推理的专门研究
2. **缓解策略缺乏**: 虽然问题已被识别，但有效的缓解策略研究仍然稀少
3. **模型架构差异**: 不同架构（decoder-only vs encoder-decoder）的比较研究不足
4. **长对话场景**: 超长对话（>20轮）的研究几乎空白

## 对后续研究的建议

1. 开发针对多轮推理的专门训练方法
2. 设计动态上下文管理机制
3. 建立标准化的多轮推理评估基准
4. 探索推理时间（inference-time）的干预策略

## 参考文献

- [P1] Smith, J. et al. "Multi-turn Reasoning Reliability in Large Language Models." ACL 2024.
- [P2] Wang, L. et al. "Context Accumulation and Attention Dilution in Conversational AI." EMNLP 2023.
- [P3] Chen, R. et al. "A Framework for Evaluating Multi-turn Logical Consistency." NeurIPS 2024.
- [P4] Zhang, Y. et al. "Reasoning Degradation in Extended LLM Conversations." ICML 2023.
- [P5] Liu, H. et al. "Dialogue History Impact on Inference Quality." ACL 2024.
- [P6] Brown, K. et al. "Attention Dynamics Across Conversation Turns." NAACL 2023.
- [P7] Davis, M. et al. "Benchmarking Multi-turn Reasoning in Language Models." TACL 2024.
- [P8] Lee, S. et al. "Context Window Optimization for Improved Reasoning." EMNLP 2023.
"""

# ---- Quick Research mock data ----

MOCK_ANSWER_SCHEMA_JSON = {
    "question_type": "comparative",
    "subject": "retrieval-augmented generation techniques",
    "comparison_target": "RAG techniques",
    "outcome": "reduction of factual hallucination",
    "required_dimensions": [
        "technique",
        "baseline",
        "task",
        "dataset",
        "metric",
        "reported_result",
        "limitations",
    ],
    "inclusion_guidance": [
        "Empirical comparisons of RAG methods",
        "Studies with quantitative hallucination metrics",
        "Papers from 2022-2026",
    ],
    "exclusion_guidance": [
        "Pure theoretical papers without empirical evaluation",
        "Non-English sources without English abstract",
    ],
}

MOCK_QUICK_QUERIES_JSON = {
    "queries": [
        {
            "query_id": "q001",
            "query": "retrieval augmented generation hallucination reduction empirical comparison 2024",
            "purpose": "empirical_comparison",
            "round_index": 0,
        },
        {
            "query_id": "q002",
            "query": "RAG techniques factual consistency evaluation benchmark dataset",
            "purpose": "benchmark",
            "round_index": 0,
        },
        {
            "query_id": "q003",
            "query": "retrieval augmented generation hallucination mitigation methods survey 2023 2024",
            "purpose": "review",
            "round_index": 0,
        },
        {
            "query_id": "q004",
            "query": "RAG reranking retrieval quality factual accuracy metric comparison",
            "purpose": "metric",
            "round_index": 0,
        },
        {
            "query_id": "q005",
            "query": "retrieval augmented generation limitations failure cases hallucination study",
            "purpose": "limitations",
            "round_index": 0,
        },
        {
            "query_id": "q006",
            "query": "grounded generation retrieval evidence LLM factuality evaluation",
            "purpose": "technique_category",
            "round_index": 0,
        },
    ],
}

MOCK_COVERAGE_JSON = {
    "sufficient": True,
    "covered_dimensions": [
        "technique",
        "baseline",
        "task",
        "dataset",
        "metric",
        "reported_result",
        "limitations",
    ],
    "missing_dimensions": [],
    "covered_techniques": [
        "Standard RAG",
        "Self-RAG",
        "CRAG",
        "REALM",
        "Atlas",
        "RAG with reranking",
    ],
    "underrepresented_areas": [],
    "source_count": 15,
    "high_quality_source_count": 12,
    "new_queries": [],
    "reason": "All required dimensions are covered with sufficient source diversity.",
}

MOCK_RESEARCH_NOTE_JSON = {
    "note_id": "note001",
    "source_id": "src001",
    "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "url": "https://arxiv.org/abs/2310.11511",
    "year": 2023,
    "source_type": "preprint",
    "research_type": "empirical",
    "technique": "Self-RAG",
    "baseline": "Standard RAG, RETRO, Atlas",
    "task": "Open-domain QA, fact verification",
    "datasets": ["PubHealth", "ARC-Challenge", "PopQA", "TriviaQA"],
    "metrics": ["accuracy", "factuality score", "citation precision"],
    "reported_results": [
        "Self-RAG outperforms standard RAG by 5-15% on factuality metrics across 6 benchmarks",
        "Self-RAG with reflection tokens achieves 78.3% on PubHealth fact verification",
    ],
    "limitations": [
        "Requires training data with reflection tokens",
        "Additional inference cost from multiple retrieval-reflection cycles",
    ],
    "relevant_quotes": [
        "Self-RAG significantly outperforms standard RAG with a 5-15% gain on factuality metrics.",
        "The self-reflection mechanism enables the model to decide when to retrieve and critique its own generations.",
    ],
    "relevance_summary": "Direct empirical comparison of Self-RAG against multiple baselines on hallucination reduction metrics.",
    "confidence": "high",
    "extraction_failed": False,
}

MOCK_COMPARISON_MATRIX_JSON = {
    "matrix": [
        {
            "technique": "Self-RAG",
            "baseline": "Standard RAG",
            "task_or_domain": "Open-domain QA",
            "datasets": ["PubHealth", "PopQA"],
            "metrics": ["factuality score", "accuracy"],
            "reported_result": "5-15% improvement in factuality vs standard RAG",
            "limitations": ["Training data requirement", "Inference cost"],
            "source_ids": ["src001", "src002"],
            "support_count": 2,
            "confidence": "high",
            "note": "",
        },
        {
            "technique": "CRAG (Corrective RAG)",
            "baseline": "Standard RAG",
            "task_or_domain": "Fact verification",
            "datasets": ["FEVER", "PubHealth"],
            "metrics": ["hallucination rate", "precision"],
            "reported_result": "Reduces hallucination rate by 8-12% on FEVER",
            "limitations": ["Dependent on retrieval quality evaluation step"],
            "source_ids": ["src003"],
            "support_count": 1,
            "confidence": "medium",
            "note": "domain-specific evidence",
        },
        {
            "technique": "RAG with Reranking",
            "baseline": "RAG without reranking",
            "task_or_domain": "Multi-document QA",
            "datasets": ["ASQA", "ALCE"],
            "metrics": ["citation recall", "factual precision"],
            "reported_result": "Reranking improves factual precision by 10-18% across datasets",
            "limitations": ["Reranker model quality dependency", "Latency increase"],
            "source_ids": ["src004", "src005"],
            "support_count": 2,
            "confidence": "high",
            "note": "",
        },
    ],
}

MOCK_QUICK_REPORT = """# 执行摘要

- **Self-RAG** 通过自反思机制在多项基准上一致减少事实幻觉，相较于标准 RAG 提升 5-15% 的事实性得分 [S1][S2]
- **检索重排序（Reranking）** 是提升检索质量的通用有效手段，能改善事实精确度 10-18% [S4][S5]
- **CRAG** 通过检索质量评估和知识校正减少幻觉率 8-12%，但其效果高度依赖检索评估步骤的准确性 [S3]
- 当前证据表明，结合检索自省与重排序的方法在减少幻觉方面表现最为一致 [S1][S4]
- 现有研究主要集中在英文开放域 QA 任务，跨语言和特定领域场景的证据仍然有限 [S6][S7]

# 研究问题与范围

## 研究问题

Which retrieval-augmented generation techniques most consistently reduce factual hallucination?

## 检索范围

- **检索轮次**: 1 轮
- **来源数**: 15 个高质量来源
- **检索关键词**: RAG, hallucination, factual consistency, retrieval augmentation, grounded generation
- **年份范围**: 2022–2026

# 主要方法比较

## Self-RAG：自反思式检索增强生成

Self-RAG 通过训练模型在生成过程中输出特殊的反思标记（reflection tokens），使模型能够判断何时需要检索、检索内容是否相关、以及生成是否得到检索结果的支持。多项研究一致表明，Self-RAG 在 PubHealth、PopQA、TriviaQA 等多个基准上显著优于标准 RAG [S1][S2]。

## CRAG：校正式检索增强生成

CRAG 引入检索质量评估器（retrieval evaluator），在检索结果质量不足时自动触发知识校正或网页搜索。在 FEVER 和 PubHealth 数据集上，CRAG 减少幻觉率 8-12% [S3]。

## 检索重排序

多个独立研究发现，在 RAG 流程中加入重排序步骤可以显著提升最终生成的事实精确度。重排序帮助过滤检索噪音，确保只有最相关的文档片段进入生成上下文 [S4][S5]。

# 哪些方法的证据最一致

基于当前检索来源，以下发现得到多个独立来源的支持：

1. **检索质量是幻觉减少的核心因素**：无论具体技术路线如何，提升检索结果的相关性和准确性是减少幻觉的前提 [S1][S4][S5]
2. **自省机制有显著增益**：在生成过程中增加对检索结果的验证和反思，一致地减少事实错误 [S1][S2][S3]
3. **重排序是低成本的通用提升手段**：在标准 RAG 流程中加入重排序模块，几乎在所有实验中都能提升事实性指标 [S4][S5]

以下发现来自单一来源，需谨慎解读：
- CRAG 在特定条件下（低质量初始检索）的效果尤为突出，但这是基于单一研究的领域特定证据 [S3]

# 适用场景与边界

当前证据主要适用于以下场景：
- 英文开放域问答
- 事实验证任务
- 有高质量检索语料库支持的场景

以下场景的证据不足：
- 多语言和跨语言场景
- 特定专业领域（如医疗、法律）
- 长文本生成中的幻觉控制

# 研究不足与未来方向

## 来源明确提出的限制

- 多数方法的额外计算开销在 20-50% 之间，但部分研究未完整报告推理延迟 [S2][S4]
- 训练数据需求限制了 Self-RAG 等方法在低资源场景的应用 [S1]

## 基于当前检索来源的有限推断

在本次检索到的来源中，以下方面尚未得到充分覆盖：
- 当前证据尚未覆盖不同 RAG 方法在对话场景中的长期可靠性
- 领域迁移（domain transfer）对幻觉控制方法的影响缺乏系统研究
- 模型规模与 RAG 策略的交互效应尚未被充分探索

# 方法与证据限制

本报告基于公开网页、论文页面、摘要及可访问正文生成。引用均指向实际检索来源，但并非所有结论均经过论文 PDF 全文逐句核验。

# 参考来源

[S1] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection — https://arxiv.org/abs/2310.11511
[S2] Benchmarking RAG Methods for Factual Consistency — https://arxiv.org/abs/2312.xxxxx
[S3] Corrective Retrieval Augmented Generation — https://arxiv.org/abs/2401.xxxxx
[S4] Reranking for Retrieval-Augmented Generation: A Comprehensive Study — https://arxiv.org/abs/2311.xxxxx
[S5] The Impact of Retrieval Quality on LLM Factuality — https://aclanthology.org/2024.xxxxx
[S6] Limitations of Current RAG Evaluation Benchmarks — https://arxiv.org/abs/2402.xxxxx
[S7] RAG in Non-English Languages: A Preliminary Study — https://arxiv.org/abs/2403.xxxxx
"""


class MockLLMClient:
    """Mock LLM client for development and testing without API keys."""

    def __init__(self):
        self.call_count = 0
        self.tokens_used = 0

    async def close(self) -> None:
        pass

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[str, dict]:
        """Return mock text based on the prompt context."""
        self.call_count += 1
        self.tokens_used += 500

        combined = (system_prompt + " " + user_prompt).lower()

        # Quick report: system prompt contains the Quick Research report markers
        if "source_of_truth" in combined and "researchnotes" in combined:
            return MOCK_QUICK_REPORT, {
                "prompt_tokens": 2000,
                "completion_tokens": 3000,
                "total_tokens": 5000,
                "model": model or "mock",
            }

        if "report" in combined or "synthesize" in combined:
            return MOCK_REPORT, {
                "prompt_tokens": 800,
                "completion_tokens": 1200,
                "total_tokens": 2000,
                "model": model or "mock",
            }

        # Default: return a short response
        return "Mock response for text generation.", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": model or "mock",
        }

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[Optional[BaseModel], dict]:
        """Return mock structured output based on the expected model type."""
        self.call_count += 1
        self.tokens_used += 800

        combined = (system_prompt + " " + user_prompt).lower()
        model_name = output_model.__name__

        usage = {
            "prompt_tokens": 300,
            "completion_tokens": 500,
            "total_tokens": 800,
            "model": model or "mock",
        }

        try:
            # Route to the right mock data based on model type
            if "SearchPlan" in model_name:
                from app.models.search_plan import SearchPlan
                return SearchPlan.model_validate(MOCK_SEARCH_PLAN_JSON), usage

            elif "GapAnalysis" in model_name:
                from app.models.evidence import GapAnalysis
                return GapAnalysis.model_validate(MOCK_GAP_ANALYSIS_JSON), usage

            elif "AnswerSchema" in model_name:
                from app.models.quick_research import AnswerSchema
                return AnswerSchema.model_validate(MOCK_ANSWER_SCHEMA_JSON), usage

            elif "CoverageAssessment" in model_name:
                from app.models.quick_research import CoverageAssessment
                return CoverageAssessment.model_validate(MOCK_COVERAGE_JSON), usage

            elif "ResearchNote" in model_name:
                from app.models.quick_research import ResearchNote
                return ResearchNote.model_validate(MOCK_RESEARCH_NOTE_JSON), usage

            elif "ComparisonRow" in model_name or "_MatrixOutput" in model_name:
                from app.models.quick_research import ComparisonRow
                matrix_data = MOCK_COMPARISON_MATRIX_JSON
                # Check if the output model has a 'matrix' field (i.e., _MatrixOutput wrapper)
                if hasattr(output_model, "model_fields") and "matrix" in output_model.model_fields:
                    return output_model.model_validate(matrix_data), usage
                # If asking for a single ComparisonRow, return first row
                if matrix_data.get("matrix"):
                    return ComparisonRow.model_validate(matrix_data["matrix"][0]), usage
                return output_model.model_validate({}), usage

            elif "_QueryList" in model_name:
                return output_model.model_validate(MOCK_QUICK_QUERIES_JSON), usage

            elif "rankings" in output_model.model_json_schema().get("properties", {}):
                return output_model.model_validate(
                    {"rankings": MOCK_RANKING_RESULTS}
                ), usage

            else:
                # Try to construct a minimal valid instance
                try:
                    instance = output_model.model_validate({})
                except Exception:
                    # Some models require non-empty fields — try with simple defaults
                    schema = output_model.model_json_schema()
                    props = schema.get("properties", {})
                    defaults = {}
                    for key, prop in props.items():
                        if prop.get("type") == "string":
                            defaults[key] = f"mock_{key}"
                        elif prop.get("type") == "array":
                            defaults[key] = []
                        elif prop.get("type") == "integer":
                            defaults[key] = 0
                        elif prop.get("type") == "number":
                            defaults[key] = 0.0
                        elif prop.get("type") == "boolean":
                            defaults[key] = False
                    instance = output_model.model_validate(defaults)
                return instance, usage

        except Exception as e:
            logger.error(f"Mock structured generation failed for {model_name}: {e}")
            return None, usage

    def _extract_usage(self, response: dict) -> dict:
        return response.get("usage", {})


def get_mock_llm_client() -> MockLLMClient:
    return MockLLMClient()
