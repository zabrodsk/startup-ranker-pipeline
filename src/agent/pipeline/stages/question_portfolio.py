"""Soft question-count guidance for investment-question decomposition."""

from __future__ import annotations

from agent.pipeline.stages.constants import QuestionAspect

QUESTION_PORTFOLIO_TOTAL = 76
# Per-category soft maxima. These bound downstream work but are never minimums.
QUESTION_PORTFOLIO_ALLOCATION: dict[QuestionAspect, int] = {
    "general_company": 14,
    "market": 24,
    "product": 20,
    "team": 18,
}


def build_portfolio_instruction(aspect: QuestionAspect, budget: int) -> str:
    """Return simple question-count guidance for one portfolio category."""
    return f"""
QUESTION PORTFOLIO GUIDANCE
Generate up to {budget} questions for the {aspect} category, including the root question.
Fewer questions are acceptable. Do not add filler merely to reach the maximum.
Prioritize questions whose answers could change the investment assessment, expose
a material risk, or test a key hypothesis. Keep questions specific and independently
answerable from company documents or credible external evidence.
""".strip()
