"""State for the iterative investment story pipeline.

This module defines the state classes for the investment analysis pipeline:
- InputState: The input schema for invoking the graph (used in langgraph.dev)
- IterativeInvestmentStoryState: The full state tracking all pipeline data
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.dataclasses.config import Config
from agent.dataclasses.question_tree import QuestionTree
from agent.dataclasses.ranking import CompanyRankingResult
from agent.dataclasses.examples import BRANDBACK_COMPANY


class InputState(BaseModel):
    """Input state for the investment analysis pipeline.

    This is what users provide when invoking the graph (e.g., in langgraph.dev).
    Only requires the company to analyze - all other fields have sensible defaults.

    When all_qa_pairs is pre-populated (e.g. from batch file-based ingestion),
    the pipeline skips decomposition and answering, jumping straight to
    argument generation.

    Attributes:
        company: The company to analyze (required)
        config: Optional pipeline configuration
        all_qa_pairs: Pre-populated Q&A pairs (skips decomposition/answering)
    """

    company: Company = BRANDBACK_COMPANY
    config: Config = Config(
        n_pro_arguments=3,
        n_contra_arguments=3,
        k_best_arguments_per_iteration=[3, 1],
        max_iterations=2,
    )
    # Values may be str, list[str] (chunk_ids), or None (web_search_query/results)
    all_qa_pairs: list[Dict[str, Any]] = Field(default_factory=list)
    scoring_signals: Dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)
    vc_context: str = ""
    slug: str = ""

    # VC matching optimisation: pre-computed from a prior full analysis.
    # When both are provided, the graph skips Stages 1-7 and enters at Stage 8 only.
    final_arguments: list[Any] = Field(default_factory=list)
    final_decision: Optional[str] = None

    # Skip the executive-summary node after composite ranking. Set by callers
    # that only consume ranking_result (e.g. VC matching) — saves one
    # reasoning-heavy LLM call per invocation.
    skip_executive_summary: bool = False


class IterativeInvestmentStoryState(BaseModel):
    """Main state tracking arguments through refinement iterations.

    The pipeline generates pro/contra arguments, applies devil's advocate
    critiques, scores them, and refines the best ones over multiple iterations.

    Attributes:
        company: The company being analyzed (primary input)
        config: Pipeline configuration (num arguments, iterations, etc.)
        question_trees: Dict of answered question trees for each aspect
        all_qa_pairs: Combined Q&A pairs from all trees for argument generation
        is_final: If True, skip generation and go straight to scoring

        current_arguments: Arguments being processed in current iteration
        refined_arguments: Arguments after refinement
        selected_arguments: Top K arguments selected for refinement
        arguments_history: History of all iterations for analysis

        final_arguments: The final set of arguments after all iterations
        final_decision: Investment recommendation (invest/not_invest)
    """
    company: Company = BRANDBACK_COMPANY
    config: Config = Config(
        n_pro_arguments=3,
        n_contra_arguments=3,
        k_best_arguments_per_iteration=[3, 1],
        max_iterations=2,
    )
    # Question trees for all 4 aspects (built during decomposition stage)
    # Keys: "general_company", "market", "product", "team"
    question_trees: Dict[str, QuestionTree] = Field(default_factory=dict)

    # Combined Q&A pairs from all trees (populated after answering stage).
    # Each dict has question/answer (str), chunk_ids (list[str]), web_search_* (str|None).
    all_qa_pairs: list[Dict[str, Any]] = Field(default_factory=list)
    scoring_signals: Dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)
    vc_context: str = ""
    slug: str = ""

    # Legacy field for backward compatibility - will be deprecated
    question_tree: QuestionTree | None = None

    # Pipeline control
    is_final: bool = False
    company_name: str = ""
    # Skip the executive-summary node after composite ranking (see InputState).
    skip_executive_summary: bool = False

    # Argument tracking
    current_arguments: list[Argument] = Field(default_factory=list)
    refined_arguments: list[Argument] = Field(default_factory=list)
    selected_arguments: list[Argument] = Field(default_factory=list)
    arguments_history: list[Dict[str, Any]] = Field(default_factory=list)
    current_iteration: int = 0

    # Per-type argument tracking
    pro_arguments: list[Argument] = Field(default_factory=list)
    contra_arguments: list[Argument] = Field(default_factory=list)
    devils_advocate_pro_arguments: list[Argument] = Field(default_factory=list)
    devils_advocate_contra_arguments: list[Argument] = Field(default_factory=list)
    refined_pro_arguments: list[Argument] = Field(default_factory=list)
    refined_contra_arguments: list[Argument] = Field(default_factory=list)

    # ===== OUTPUT FIELDS =====
    final_arguments: list[Argument] = Field(default_factory=list)
    final_decision: Literal["invest", "not_invest"] | None = None
    ranking_result: CompanyRankingResult | None = None

    @property
    def should_continue_iterations(self) -> bool:
        """Check if we should continue with more iterations."""
        return self.current_iteration < self.config.max_iterations

    def get_current_k_best(self) -> int:
        """Get k_best_arguments for the current iteration."""
        return self.config.get_k_best_for_iteration(self.current_iteration)
