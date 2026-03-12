import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.stages import parallel_decomposition


def test_parallel_decomposition_uses_overridden_root_questions(monkeypatch):
    seen_questions = {}

    async def fake_get_or_decompose_question(
        question: str,
        industry: str,
        aspect: str,
        company_name: str,
        prompt_overrides: dict | None = None,
    ):
        seen_questions[aspect] = question
        return {
            "aspect": aspect,
            "tree": QuestionTree(aspect=aspect, root_node=QuestionNode(question=question)),
        }

    monkeypatch.setattr(
        parallel_decomposition,
        "_get_or_decompose_question",
        fake_get_or_decompose_question,
    )

    state = IterativeInvestmentStoryState(
        company=Company(name="Acme", industry="Fintech"),
        prompt_overrides={
            "values": {
                "questions.market": "Custom market root question?",
            }
        },
    )

    result = asyncio.run(parallel_decomposition.decompose_all_questions(state))

    assert seen_questions["market"] == "Custom market root question?"
    assert result["question_trees"]["market"].root_node.question == "Custom market root question?"


def test_parallel_decomposition_cache_remains_company_scoped(monkeypatch):
    seen = {}
    cached_tree = QuestionTree(
        aspect="market",
        root_node=QuestionNode(question="Cached market root?"),
    )

    def fake_get_cached_question_tree(question, company, aspect):
        seen["question"] = question
        seen["company_name"] = company.name
        seen["industry"] = company.industry
        seen["aspect"] = aspect
        return cached_tree

    monkeypatch.setattr(
        parallel_decomposition,
        "get_cached_question_tree",
        fake_get_cached_question_tree,
    )

    result = asyncio.run(
        parallel_decomposition._get_or_decompose_question(
            question="What is the market size?",
            industry="Vertical SaaS / Hospitality",
            aspect="market",
            company_name="Apaleo",
            prompt_overrides={},
        )
    )

    assert seen["company_name"] == "Apaleo"
    assert seen["industry"] == "Vertical SaaS / Hospitality"
    assert result["tree"] == cached_tree
