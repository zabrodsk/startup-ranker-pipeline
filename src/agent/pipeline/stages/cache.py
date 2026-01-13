"""Caching utilities for question trees.

This module provides caching for both decomposed and answered question trees
to avoid redundant LLM calls and web searches.

Two types of caches:
- Decomposed trees: Cached after question decomposition
- Answered trees: Cached after answering (includes web search results)
"""

from typing import Any, Dict, Optional

from agent.common.cache import get, set
from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionTree
from agent.pipeline.stages.constants import QuestionAspect

# Cache file names
CACHE_NAME = "question_trees.json"
ANSWERED_CACHE_NAME = "answered_question_trees.json"


# =============================================================================
# Decomposed Question Tree Cache
# =============================================================================


def get_cache_key(question: str, company: Company) -> str:
    """Generate a cache key for a question-company pair."""
    return f"{question}_{company.name}"


def get_cached_question_tree(
    question: str, company: Company, aspect: QuestionAspect
) -> Optional[QuestionTree]:
    """Return cached QuestionTree if present, else None.

    Args:
        question: The question text
        company: The company being analyzed
        aspect: The aspect of analysis (general_company, market, product, team)

    Returns:
        Cached QuestionTree if found, None otherwise
    """
    cache_key = get_cache_key(question, company)
    cached = get(cache_key, CACHE_NAME)
    if cached is not None:
        return QuestionTree(**cached)
    return None


def cache_question_tree(
    question: str, company: Company, tree: QuestionTree
) -> None:
    """Cache a QuestionTree for later retrieval.

    Args:
        question: The question text
        company: The company being analyzed
        tree: The decomposed question tree to cache
    """
    cache_key = get_cache_key(question, company)
    question_tree_dict: Dict[str, Any] = tree.model_dump()
    set(cache_key, question_tree_dict, CACHE_NAME)


# =============================================================================
# Answered Question Tree Cache
# =============================================================================


def get_answered_cache_key(aspect: str, company: Company) -> str:
    """Generate a cache key for an answered question tree."""
    return f"answered_{aspect}_{company.name}"


def get_cached_answered_tree(
    aspect: QuestionAspect, company: Company
) -> Optional[QuestionTree]:
    """Return cached answered QuestionTree if present, else None.

    Args:
        aspect: The question aspect (general_company, market, product, team)
        company: The company being analyzed

    Returns:
        Cached answered QuestionTree if found, None otherwise
    """
    cache_key = get_answered_cache_key(aspect, company)
    cached = get(cache_key, ANSWERED_CACHE_NAME)
    if cached is not None:
        return QuestionTree(**cached)
    return None


def cache_answered_tree(
    aspect: QuestionAspect, company: Company, tree: QuestionTree
) -> None:
    """Cache an answered QuestionTree for later retrieval.

    Args:
        aspect: The question aspect (general_company, market, product, team)
        company: The company being analyzed
        tree: The answered question tree to cache
    """
    cache_key = get_answered_cache_key(aspect, company)
    tree_dict: Dict[str, Any] = tree.model_dump()
    set(cache_key, tree_dict, ANSWERED_CACHE_NAME)
