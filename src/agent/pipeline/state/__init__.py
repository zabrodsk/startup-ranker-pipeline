"""State classes for the investment pipeline.

This module exports all state classes used throughout the pipeline:
- IterativeInvestmentStoryState: Main pipeline state
- AnswerState: State for answering with tool support
- AnswerStateSimple: State for answering without tools
- AnswerQuestionTreeState: State for tree-wide answering
- Decomposition states: Input/output for question decomposition
- Various LLM output schemas
"""

from agent.pipeline.state.answer import (
    AnswerQuestionTreeState,
    AnswerState,
    AnswerStateSimple,
)
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionNode,
    DecompositionOutput,
    DecompositionTree,
)
from agent.pipeline.state.investment_story import (
    InputState,
    IterativeInvestmentStoryState,
)
from agent.pipeline.state.schemas import (
    ArgumentCritique,
    ArgumentOutput,
    ArgumentsOutput,
    CriterionScore,
    PersonClaim,
    PersonClaimEvidence,
    PersonProfileOutput,
    PersonProfileSections,
    PersonProvenanceRecord,
    PersonSubject,
    IndividualRefinedArgumentOutput,
    SingleArgumentScore,
)

__all__ = [
    # Input state (for langgraph.dev)
    "InputState",
    # Main state
    "IterativeInvestmentStoryState",
    # Answer states
    "AnswerState",
    "AnswerStateSimple",
    "AnswerQuestionTreeState",
    # Decomposition states
    "DecompositionInput",
    "DecompositionOutput",
    "DecompositionNode",
    "DecompositionTree",
    # LLM output schemas
    "ArgumentOutput",
    "ArgumentsOutput",
    "CriterionScore",
    "SingleArgumentScore",
    "ArgumentCritique",
    "IndividualRefinedArgumentOutput",
    "PersonSubject",
    "PersonClaimEvidence",
    "PersonProfileSections",
    "PersonClaim",
    "PersonProvenanceRecord",
    "PersonProfileOutput",
]
