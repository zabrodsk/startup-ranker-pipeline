"""Pipeline stages for investment analysis.

This module exports all stage functions and subgraphs:

Stage 1 - Decomposition: Break down complex questions (parallel)
Stage 2 - Answering: Answer questions via web search or synthesis (parallel)
Stage 3 - Generation: Generate pro/contra arguments
Stage 4 - Critique: Apply devil's advocate critiques
Stage 5 - Evaluation: Score arguments on 14 criteria
Stage 6 - Refinement: Improve arguments based on feedback
Stage 7 - Decision: Make final investment recommendation
"""

# Stage 1: Decomposition (single question)
from agent.pipeline.stages.decomposition import graph as decomposition_graph

# Stage 1b: Parallel decomposition (all 4 questions)
from agent.pipeline.stages.parallel_decomposition import decompose_all_questions

# Stage 2: Answering (single tree)
from agent.pipeline.stages.answering import (
    answer_tree_graph,
    answer_with_tool_graph,
    answer_without_tool_graph,
)

# Stage 2b: Parallel answering (all 4 trees)
from agent.pipeline.stages.parallel_answering import answer_all_trees

# Stage 3: Generation
from agent.pipeline.stages.generation import (
    check_if_final,
    generate_contra_arguments,
    generate_pro_and_contra_arguments,
    generate_pro_arguments,
    merge_arguments,
)

# Stage 4: Critique
from agent.pipeline.stages.critique import (
    apply_devils_advocate,
    apply_devils_advocate_to_contra_arguments,
    apply_devils_advocate_to_pro_arguments,
)

# Stage 5: Evaluation
from agent.pipeline.stages.evaluation import (
    score_and_select_best_k,
    score_arguments_in_parallel,
    score_single_argument,
)

# Stage 6: Refinement
from agent.pipeline.stages.refinement import (
    merge_refined_arguments,
    refine_contra_arguments,
    refine_pro_arguments,
)

# Stage 7: Decision
from agent.pipeline.stages.decision import (
    add_arguments_to_history,
    check_continue,
    create_final_investment_story,
    decide_final_investment_decision,
    prepare_final_arguments,
    reset_arguments_and_increment_iteration,
)

# Constants: Question definitions and types
from agent.pipeline.stages.constants import (
    INVESTMENT_QUESTIONS,
    QuestionAspect,
    get_all_aspects,
    get_question_for_aspect,
)

# Cache: Question tree caching utilities
from agent.pipeline.stages.cache import (
    cache_answered_tree,
    cache_question_tree,
    get_cached_answered_tree,
    get_cached_question_tree,
)

__all__ = [
    # Stage 1 - Decomposition
    "decomposition_graph",
    "decompose_all_questions",
    # Stage 2 - Answering
    "answer_tree_graph",
    "answer_with_tool_graph",
    "answer_without_tool_graph",
    "answer_all_trees",
    # Stage 3 - Generation
    "check_if_final",
    "generate_pro_and_contra_arguments",
    "generate_pro_arguments",
    "generate_contra_arguments",
    "merge_arguments",
    # Stage 4 - Critique
    "apply_devils_advocate",
    "apply_devils_advocate_to_pro_arguments",
    "apply_devils_advocate_to_contra_arguments",
    # Stage 5 - Evaluation
    "score_single_argument",
    "score_arguments_in_parallel",
    "score_and_select_best_k",
    # Stage 6 - Refinement
    "refine_pro_arguments",
    "refine_contra_arguments",
    "merge_refined_arguments",
    # Stage 7 - Decision
    "add_arguments_to_history",
    "reset_arguments_and_increment_iteration",
    "check_continue",
    "prepare_final_arguments",
    "decide_final_investment_decision",
    "create_final_investment_story",
    # Constants - Question definitions
    "INVESTMENT_QUESTIONS",
    "QuestionAspect",
    "get_question_for_aspect",
    "get_all_aspects",
    # Cache - Question tree caching
    "get_cached_question_tree",
    "cache_question_tree",
    "get_cached_answered_tree",
    "cache_answered_tree",
]
