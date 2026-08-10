"""State classes for question decomposition.

These state classes are used by the decomposition stage to break down
complex investment questions into hierarchical question trees.
"""

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from agent.dataclasses.question_tree import QuestionTree


class DecompositionNode(BaseModel):
    """A node in the decomposition tree from LLM output."""

    question: str
    sub_questions: list[str]
    # Web-evidence route tag (see agent.web_search.planner.QuestionRoute).
    # Optional so old cached trees and tag-less LLM outputs still parse.
    route: str | None = None
    # Legacy portfolio-selection metadata. Retained for old caches and model
    # outputs; budgeted pipeline calls do not require or validate these fields.
    coverage_tags: list[str] = Field(default_factory=list)
    decision_rationale: str | None = None
    priority: Literal["core", "supporting"] | None = None


class DecompositionTree(BaseModel):
    """The full decomposition tree from the LLM.

    Example structure:
    [
        {
            "question": "Main question",
            "sub_questions": ["Sub Q1", "Sub Q2"],
        },
        {
            "question": "Sub Q1",
            "sub_questions": ["Sub Q1a", "Sub Q1b"],
        }
    ]
    """

    nodes: list[DecompositionNode]


class DecompositionInput(BaseModel):
    """Input state for question decomposition."""

    industry: str | None = "AI marketing tools"
    question: str | None = (
        "What is the current size and forecast growth of the target market?"
    )
    aspect: Literal["general_company", "market", "product", "team"] | None = (
        "general_company"
    )
    # Soft maximum for this category, root included. Fewer usable questions are
    # accepted and excess questions are capped. None preserves legacy behavior.
    question_budget: int | None = None
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)


class DecompositionOutput(BaseModel):
    """Output state with the decomposed question tree."""

    question_tree: QuestionTree
    original_question: str
