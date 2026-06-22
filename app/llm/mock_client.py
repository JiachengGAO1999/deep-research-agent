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

        if "report" in combined or "synthesize" in combined:
            return MOCK_REPORT, {"prompt_tokens": 800, "completion_tokens": 1200, "total_tokens": 2000}

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

            elif "rankings" in output_model.model_json_schema().get("properties", {}):
                return output_model.model_validate(
                    {"rankings": MOCK_RANKING_RESULTS}
                ), usage

            else:
                # Try to construct a minimal valid instance
                schema = output_model.model_json_schema()
                # Create a simple mock instance from schema defaults
                instance = output_model.model_validate({})
                return instance, usage

        except Exception as e:
            logger.error(f"Mock structured generation failed for {model_name}: {e}")
            return None, usage

    def _extract_usage(self, response: dict) -> dict:
        return response.get("usage", {})


def get_mock_llm_client() -> MockLLMClient:
    return MockLLMClient()
