"""Mock provider that returns realistic fake papers for testing and demos."""

from __future__ import annotations

import logging
from typing import Optional

from app.models.paper import Paper, PaperSource, AuthorInfo
from app.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Pre-built mock papers covering the research domain
MOCK_PAPERS = [
    Paper(
        internal_id="mock_p1",
        title="Multi-turn Reasoning Reliability in Large Language Models",
        abstract="This paper presents a comprehensive study of how multi-turn dialogue affects the reasoning capabilities of large language models. We evaluate several popular LLMs across varying conversation lengths and find that reasoning accuracy degrades by an average of 15% after 5+ turns. We propose the Multi-turn Reasoning Consistency (MRC) metric to quantify this degradation and analyze the underlying mechanisms including attention dilution and context interference. Our results show that models with larger context windows are more susceptible to reasoning degradation in extended dialogues.",
        authors=[
            AuthorInfo(name="James Smith", affiliation="Stanford University"),
            AuthorInfo(name="Maria Garcia", affiliation="MIT"),
            AuthorInfo(name="Yuki Tanaka", affiliation="University of Tokyo"),
        ],
        publication_year=2024,
        venue="Proceedings of ACL 2024",
        doi="10.1234/acl2024.mrc.001",
        citation_count=42,
        source_names=["openalex", "semantic_scholar"],
        source_ids=[
            PaperSource(provider="openalex", provider_id="W4200000001"),
            PaperSource(provider="semantic_scholar", provider_id="paper1"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p2",
        title="Context Accumulation and Attention Dilution in Conversational AI",
        abstract="As conversational AI systems engage in increasingly longer dialogues, the accumulation of context poses challenges for maintaining reasoning quality. We investigate how attention mechanisms in transformer-based language models handle growing dialogue histories. Through controlled experiments with GPT-4, Claude, and Llama-2, we demonstrate that attention scores become progressively diluted across earlier conversation turns as dialogue length increases. This attention dilution correlates strongly with degraded reasoning performance on standardized benchmarks. We propose a context pruning method that preserves critical information while reducing the effective context length by 40%, showing improved reasoning consistency in multi-turn settings.",
        authors=[
            AuthorInfo(name="Li Wei Wang", affiliation="Carnegie Mellon University"),
            AuthorInfo(name="Ahmed Hassan", affiliation="Google Research"),
        ],
        publication_year=2023,
        venue="Proceedings of EMNLP 2023",
        doi="10.1234/emnlp2023.cad.002",
        citation_count=128,
        source_names=["semantic_scholar", "openalex"],
        source_ids=[
            PaperSource(provider="semantic_scholar", provider_id="paper2"),
            PaperSource(provider="openalex", provider_id="W4200000002"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p3",
        title="A Framework for Evaluating Multi-turn Logical Consistency in Language Models",
        abstract="We introduce LogicCheck, a comprehensive framework for evaluating the logical consistency of language model outputs across multiple conversation turns. The framework includes 500 test scenarios spanning deductive reasoning, inductive reasoning, and analogical reasoning in dialogue contexts. We benchmark 12 state-of-the-art models and find significant variance in their ability to maintain logical coherence. Our analysis reveals that chain-of-thought prompting improves single-turn reasoning but does not necessarily prevent multi-turn consistency degradation. We release LogicCheck as an open-source tool for the research community.",
        authors=[
            AuthorInfo(name="Rachel Chen", affiliation="DeepMind"),
            AuthorInfo(name="Thomas Mueller", affiliation="ETH Zurich"),
            AuthorInfo(name="Sarah Johnson", affiliation="UC Berkeley"),
        ],
        publication_year=2024,
        venue="Advances in NeurIPS 2024",
        doi="10.1234/neurips2024.logic.003",
        citation_count=18,
        source_names=["arxiv", "openalex"],
        source_ids=[
            PaperSource(provider="arxiv", provider_id="2401.12345"),
            PaperSource(provider="openalex", provider_id="W4200000003"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p4",
        title="Reasoning Degradation in Extended LLM Conversations",
        abstract="We conduct a systematic analysis of reasoning degradation patterns in extended conversations with large language models. Using a carefully designed experimental protocol with 10 popular LLMs and over 10,000 multi-turn interactions, we identify three primary degradation patterns: progressive simplification, topic drift, and logical contradiction. We find that degradation begins as early as turn 3 for some models, while others maintain reasonable performance through turn 8-10. Model scale shows a complex, non-monotonic relationship with degradation resistance. We provide recommendations for monitoring and mitigating reasoning decay in production conversational systems.",
        authors=[
            AuthorInfo(name="Yuan Zhang", affiliation="Tsinghua University"),
            AuthorInfo(name="Emily Brown", affiliation="University of Cambridge"),
            AuthorInfo(name="Park Junho", affiliation="KAIST"),
        ],
        publication_year=2023,
        venue="Proceedings of ICML 2023",
        doi="10.1234/icml2023.rd.004",
        citation_count=67,
        source_names=["semantic_scholar"],
        source_ids=[
            PaperSource(provider="semantic_scholar", provider_id="paper4"),
        ],
        open_access=False,
    ),
    Paper(
        internal_id="mock_p5",
        title="Dialogue History Impact on Inference Quality in LLMs",
        abstract="This paper investigates how the quality of inference changes as dialogue history accumulates. We define inference quality along dimensions of factual accuracy, logical coherence, and response relevance. Through experiments with varying dialogue lengths (2-20 turns) and different history representation methods, we find that: (1) inference quality degrades non-linearly with dialogue length, (2) selective history summarization outperforms full history retention for longer dialogues, and (3) the optimal history handling strategy depends on the task type. We propose a dynamic history management approach that adaptively selects between full history, summarized history, and recency-weighted history.",
        authors=[
            AuthorInfo(name="Hao Liu", affiliation="Microsoft Research"),
            AuthorInfo(name="Anna Kowalski", affiliation="University of Warsaw"),
            AuthorInfo(name="David Kim", affiliation="Seoul National University"),
        ],
        publication_year=2024,
        venue="Proceedings of ACL 2024",
        doi="10.1234/acl2024.dhi.005",
        citation_count=31,
        source_names=["openalex", "crossref"],
        source_ids=[
            PaperSource(provider="openalex", provider_id="W4200000005"),
            PaperSource(provider="crossref", provider_id="10.1234/acl2024.dhi.005"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p6",
        title="Attention Dynamics Across Conversation Turns in Transformer Models",
        abstract="We analyze the dynamics of attention patterns in transformer-based language models across multiple conversation turns. Using attention visualization and quantitative metrics, we characterize how attention distributions evolve as dialogue length increases. Key findings include: (1) self-attention heads become increasingly uniform in later turns, (2) cross-turn attention to earlier utterances decreases exponentially with turn distance, and (3) models exhibit 'attention collapse' in very long dialogues where the majority of attention is allocated to the most recent turns. We discuss implications for model architecture design and propose attention-gating mechanisms to preserve long-range reasoning capability.",
        authors=[
            AuthorInfo(name="Kevin Brown", affiliation="Allen Institute for AI"),
            AuthorInfo(name="Lisa Park", affiliation="University of Washington"),
        ],
        publication_year=2023,
        venue="Proceedings of NAACL 2023",
        doi="10.1234/naacl2023.ad.006",
        citation_count=45,
        source_names=["semantic_scholar", "arxiv"],
        source_ids=[
            PaperSource(provider="semantic_scholar", provider_id="paper6"),
            PaperSource(provider="arxiv", provider_id="2305.67890"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p7",
        title="Benchmarking Multi-turn Reasoning in Language Models: The MT-ReasonEval Dataset",
        abstract="We present MT-ReasonEval, a large-scale benchmark dataset specifically designed to evaluate multi-turn reasoning capabilities in language models. The dataset contains 2,500 multi-turn scenarios across five reasoning categories: mathematical, logical, causal, analogical, and commonsense reasoning. Each scenario includes 3-10 turns of interaction with ground-truth reasoning chains. We evaluate 15 commercial and open-source models, revealing significant gaps between single-turn and multi-turn reasoning performance. Our analysis shows that even the strongest models struggle with maintaining consistent reasoning chains beyond 5 turns, with accuracy dropping from 78% to 52% on average.",
        authors=[
            AuthorInfo(name="Michelle Davis", affiliation="Meta AI"),
            AuthorInfo(name="Robert Taylor", affiliation="University of Toronto"),
            AuthorInfo(name="Wei Chen", affiliation="Peking University"),
            AuthorInfo(name="Alex Ivanov", affiliation="Yandex Research"),
        ],
        publication_year=2024,
        venue="Transactions of the ACL (TACL) 2024",
        doi="10.1234/tacl2024.mtre.007",
        citation_count=89,
        source_names=["arxiv", "semantic_scholar", "openalex"],
        source_ids=[
            PaperSource(provider="arxiv", provider_id="2402.11111"),
            PaperSource(provider="semantic_scholar", provider_id="paper7"),
            PaperSource(provider="openalex", provider_id="W4200000007"),
        ],
        open_access=True,
    ),
    Paper(
        internal_id="mock_p8",
        title="Context Window Optimization for Improved Multi-turn Reasoning",
        abstract="We explore methods for optimizing context window utilization to improve multi-turn reasoning in large language models. Through systematic experiments, we evaluate context compression, sliding window approaches, retrieval-augmented context, and hierarchical context representation. Our results demonstrate that hybrid approaches combining compression and selective retrieval achieve the best trade-off between reasoning quality and computational efficiency. We propose a lightweight context management module that can be integrated with existing LLM deployments without retraining, reducing reasoning degradation in long conversations by 23% while decreasing API costs by 35%.",
        authors=[
            AuthorInfo(name="Sang-min Lee", affiliation="NAVER AI Lab"),
            AuthorInfo(name="Peter Anderson", affiliation="University of Edinburgh"),
        ],
        publication_year=2023,
        venue="Proceedings of EMNLP 2023",
        doi="10.1234/emnlp2023.cwo.008",
        citation_count=52,
        source_names=["openalex"],
        source_ids=[
            PaperSource(provider="openalex", provider_id="W4200000008"),
        ],
        open_access=False,
    ),
    Paper(
        internal_id="mock_p9",
        title="Single-Turn Reasoning Benchmarks: A Comprehensive Survey",
        abstract="This survey paper provides a comprehensive overview of single-turn reasoning benchmarks for large language models. We categorize benchmarks across mathematical reasoning, logical reasoning, commonsense reasoning, and symbolic reasoning. While primarily focused on single-turn evaluation, we briefly discuss the limitations of current benchmarks for assessing multi-turn reasoning capabilities and identify directions for future benchmark development.",
        authors=[
            AuthorInfo(name="John Miller", affiliation="University of Oxford"),
            AuthorInfo(name="Fatima Zahra", affiliation="MBZUAI"),
        ],
        publication_year=2022,
        venue="ACM Computing Surveys",
        doi="10.1234/acmcs2022.str.009",
        citation_count=215,
        source_names=["crossref"],
        source_ids=[
            PaperSource(provider="crossref", provider_id="10.1234/acmcs2022.str.009"),
        ],
        open_access=False,
    ),
    Paper(
        internal_id="mock_p10",
        title="Scaling Laws for Conversational AI Systems",
        abstract="We study the scaling behavior of conversational AI systems as a function of model size, training data, and dialogue length. Our findings indicate that while larger models generally perform better on multi-turn tasks, the marginal benefit of scaling diminishes for conversations exceeding 10 turns. We also identify computational bottlenecks in the inference pipeline for long-context processing and propose efficient approximation methods.",
        authors=[
            AuthorInfo(name="Carlos Rodriguez", affiliation="Anthropic"),
            AuthorInfo(name="Nina Petrov", affiliation="Moscow State University"),
        ],
        publication_year=2024,
        venue="arXiv preprint",
        doi=None,
        citation_count=3,
        source_names=["arxiv"],
        source_ids=[
            PaperSource(provider="arxiv", provider_id="2403.99999"),
        ],
        open_access=True,
    ),
]

# Duplicate-like papers for dedup testing
MOCK_DUPLICATE_PAPER = Paper(
    internal_id="mock_p2_dup",
    title="Context Accumulation and Attention Dilution in Conversational  AI",  # Extra space
    abstract="A shorter abstract from a different source.",
    authors=[
        AuthorInfo(name="L. W. Wang", affiliation="CMU"),  # Different author format
        AuthorInfo(name="A. Hassan", affiliation="Google"),
    ],
    publication_year=2023,
    doi="10.1234/emnlp2023.cad.002",  # Same DOI as mock_p2
    source_names=["crossref"],
    source_ids=[PaperSource(provider="crossref", provider_id="10.1234/emnlp2023.cad.002")],
    open_access=False,
)


class MockProvider(BaseProvider):
    """Mock provider that returns pre-built papers for testing."""

    name = "mock"

    def __init__(self, settings=None, papers: Optional[list[Paper]] = None):
        super().__init__(settings)
        self._papers = papers or MOCK_PAPERS

    async def is_available(self) -> bool:
        return True

    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None
    ) -> list[Paper]:
        """Return mock papers filtered by query and year."""
        query_lower = query.lower()
        results = []

        for paper in self._papers:
            # Simple keyword matching
            title_lower = paper.title.lower()
            abstract_lower = (paper.abstract or "").lower()
            if any(
                term in title_lower or term in abstract_lower
                for term in query_lower.split()
                if len(term) > 2
            ):
                # Year filter
                if year_from and paper.publication_year and paper.publication_year < year_from:
                    continue
                if year_to and paper.publication_year and paper.publication_year > year_to:
                    continue
                results.append(paper)

        if not results:
            # Return all if nothing matches (for broader coverage)
            results = self._papers[:5]

        logger.info(f"{self.name}: found {len(results)} results for query '{query[:60]}...'")
        return results
