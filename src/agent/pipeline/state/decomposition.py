"""State classes for question decomposition.

These state classes are used by the decomposition stage to break down
complex investment questions into hierarchical question trees.
"""

from typing import Literal

from pydantic import BaseModel

from agent.dataclasses.question_tree import QuestionTree


class DecompositionNode(BaseModel):
    """A node in the decomposition tree from LLM output."""

    question: str
    sub_questions: list[str]


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
    question: str | None = "What is the current size and forecast growth of the target market?"
    aspect: Literal["general_company", "market", "product", "team"] | None = "general_company"


class DecompositionOutput(BaseModel):
    """Output state with the decomposed question tree."""

    question_tree: QuestionTree
    original_question: str
