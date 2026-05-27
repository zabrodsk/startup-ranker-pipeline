"""Tests for incremental Q&A enrichment used by re-evaluation.

Verifies the three key contracts from the plan:

1. If no new chunk is TF-IDF relevant to a question, the previous answer is
   preserved verbatim (no LLM call).
2. If a new chunk is TF-IDF relevant, the LLM is called and the enriched
   answer replaces the previous one while the new chunk's id is unioned into
   `chunk_ids`.
3. Cross-question isolation: a chunk relevant only to question A does NOT
   trigger an LLM call for question B.

We mock the LLM at module level so no API keys are required.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def patched_enrichment(monkeypatch):
    """Import evidence_answering and patch create_llm + stage context helpers."""
    from agent import evidence_answering as ea
    from agent.dataclasses.company import Company
    from agent.dataclasses.question_tree import QuestionNode, QuestionTree
    from agent.ingest.store import Chunk

    calls: list[dict] = []

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeLLM:
        async def ainvoke(self, messages):  # noqa: ANN001
            # Record the call and return a deterministic enriched answer.
            user_content = ""
            for m in messages:
                if getattr(m, "__class__", None).__name__ == "HumanMessage":
                    user_content = getattr(m, "content", "") or ""
            calls.append({"user": user_content})
            return _FakeResponse(
                "ENRICHED: Apaleo is a cloud-native hotel PMS serving 500+ "
                "properties across Europe [chunk_new_1]."
            )

    def _fake_create_llm(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeLLM()

    # Patch the LLM factory and the context managers so no real pipeline
    # policy / collector is required.
    from contextlib import contextmanager

    @contextmanager
    def _noop_ctx(*args, **kwargs):  # noqa: ANN001, ARG001
        yield

    monkeypatch.setattr(ea, "create_llm", _fake_create_llm)
    monkeypatch.setattr(ea, "use_phase_llm", _noop_ctx)
    monkeypatch.setattr(ea, "use_stage_context", _noop_ctx)
    monkeypatch.setattr(ea, "get_current_pipeline_policy", lambda: None)

    return SimpleNamespace(
        ea=ea,
        Company=Company,
        QuestionNode=QuestionNode,
        QuestionTree=QuestionTree,
        Chunk=Chunk,
        calls=calls,
    )


def _build_tree(question: str, answer: str, aspect: str = "general") -> "QuestionTree":  # noqa: F821
    from agent.dataclasses.question_tree import QuestionNode, QuestionTree

    node = QuestionNode(question=question, answer=answer, aspect=aspect, sub_nodes=[])
    return QuestionTree(aspect=aspect, root_node=node)


def test_enrichment_preserves_answer_when_no_relevant_chunks(patched_enrichment):
    p = patched_enrichment
    ea = p.ea

    previous_qa_pairs = [
        {
            "question": "What is the total addressable market for Apaleo?",
            "answer": "Apaleo targets hotels with 25-500 rooms, a ~$3B TAM.",
            "aspect": "general",
            "chunk_ids": ["chunk_1", "chunk_2"],
            "chunks_preview": "[chunk_1]: ...",
            "web_search_decision": "not needed",
        }
    ]
    question_trees = {
        "general": _build_tree(
            "What is the total addressable market for Apaleo?",
            "Apaleo targets hotels with 25-500 rooms, a ~$3B TAM.",
        )
    }
    # New chunk is about something completely unrelated.
    new_chunks = [
        p.Chunk(
            chunk_id="chunk_new_1",
            text="Recipe for chocolate chip cookies: butter, sugar, flour, eggs.",
            source_file="cookies.md",
            page_or_slide="1",
        )
    ]
    company = p.Company(name="Apaleo", industry="Hospitality software", tagline="", about="")

    result = asyncio.run(
        ea.enrich_qa_pairs_with_new_evidence(
            previous_qa_pairs=previous_qa_pairs,
            question_trees=question_trees,
            new_chunks=new_chunks,
            company=company,
        )
    )

    assert len(result) == 1
    assert result[0]["answer"] == "Apaleo targets hotels with 25-500 rooms, a ~$3B TAM."
    assert result[0]["chunk_ids"] == ["chunk_1", "chunk_2"]
    # LLM must not have been called for the irrelevant chunk.
    assert p.calls == []


def test_enrichment_calls_llm_and_unions_chunk_ids_when_relevant(patched_enrichment):
    p = patched_enrichment
    ea = p.ea

    previous_qa_pairs = [
        {
            "question": "What is Apaleo's product and which customer segment does it serve?",
            "answer": "Apaleo is a hotel PMS for small hotels.",
            "aspect": "general",
            "chunk_ids": ["chunk_1"],
            "chunks_preview": "[chunk_1]: ...",
            "web_search_decision": "not needed",
        }
    ]
    question_trees = {
        "general": _build_tree(
            "What is Apaleo's product and which customer segment does it serve?",
            "Apaleo is a hotel PMS for small hotels.",
        )
    }
    new_chunks = [
        p.Chunk(
            chunk_id="chunk_new_1",
            text=(
                "Apaleo is a cloud-native hotel PMS serving 500+ properties across "
                "Europe. Customer segment: independent hotel groups and serviced "
                "apartments. Product is fully API-first."
            ),
            source_file="Apaleo 88b60ef1.md",
            page_or_slide="1",
        )
    ]
    company = p.Company(name="Apaleo", industry="Hospitality software", tagline="", about="")

    result = asyncio.run(
        ea.enrich_qa_pairs_with_new_evidence(
            previous_qa_pairs=previous_qa_pairs,
            question_trees=question_trees,
            new_chunks=new_chunks,
            company=company,
        )
    )

    assert len(result) == 1
    updated = result[0]
    assert updated["answer"].startswith("ENRICHED:")
    assert "chunk_1" in updated["chunk_ids"]
    assert "chunk_new_1" in updated["chunk_ids"]
    # chunks_preview extended with new chunk text.
    assert "chunk_new_1" in (updated.get("chunks_preview") or "")
    # LLM must have been called exactly once.
    assert len(p.calls) == 1
    # The LLM prompt should have included the previous answer and the new chunk text.
    user_text = p.calls[0]["user"]
    assert "hotel PMS for small hotels" in user_text
    assert "cloud-native hotel PMS serving 500+" in user_text


def test_enrichment_cross_question_isolation(patched_enrichment):
    """A new chunk relevant only to one question must not trigger LLM for others."""
    p = patched_enrichment
    ea = p.ea

    previous_qa_pairs = [
        {
            "question": "What is Apaleo's product and customer segment?",
            "answer": "Apaleo is a hotel PMS.",
            "aspect": "general",
            "chunk_ids": ["chunk_a"],
        },
        {
            "question": "Who are the founding team members and their backgrounds?",
            "answer": "Founded by Uli Pillau and Martin Reichenbach in Munich in 2014.",
            "aspect": "team",
            "chunk_ids": ["chunk_b"],
        },
    ]
    question_trees = {
        "general": _build_tree(
            "What is Apaleo's product and customer segment?",
            "Apaleo is a hotel PMS.",
            aspect="general",
        ),
        "team": _build_tree(
            "Who are the founding team members and their backgrounds?",
            "Founded by Uli Pillau and Martin Reichenbach in Munich in 2014.",
            aspect="team",
        ),
    }
    # Chunk heavily mentions product/customer — lexically relevant to Q1 only.
    new_chunks = [
        p.Chunk(
            chunk_id="chunk_new_1",
            text=(
                "Apaleo product serves customer segment of 500 independent hotels "
                "across Europe; product is fully cloud-native API-first hotel PMS. "
                "Customer integrations number 200+."
            ),
            source_file="prod.md",
            page_or_slide="1",
        )
    ]
    company = p.Company(name="Apaleo", industry="Hospitality software", tagline="", about="")

    result = asyncio.run(
        ea.enrich_qa_pairs_with_new_evidence(
            previous_qa_pairs=previous_qa_pairs,
            question_trees=question_trees,
            new_chunks=new_chunks,
            company=company,
        )
    )

    assert len(result) == 2
    # Q1 answer was enriched.
    assert result[0]["answer"].startswith("ENRICHED:")
    assert "chunk_new_1" in result[0]["chunk_ids"]
    # Q2 (team) was preserved verbatim.
    assert result[1]["answer"].startswith("Founded by Uli Pillau")
    assert result[1]["chunk_ids"] == ["chunk_b"]
    # Exactly one LLM call (for Q1 only).
    assert len(p.calls) == 1


def test_enrichment_fallback_to_previous_on_llm_error(patched_enrichment, monkeypatch):
    """If the LLM raises, the enrichment path must return the previous answer."""
    p = patched_enrichment
    ea = p.ea

    class _RaisingLLM:
        async def ainvoke(self, messages):  # noqa: ANN001, ARG002
            raise RuntimeError("simulated API failure")

    monkeypatch.setattr(ea, "create_llm", lambda *a, **kw: _RaisingLLM())

    previous_qa_pairs = [
        {
            "question": "What is Apaleo's product?",
            "answer": "Apaleo is a hotel PMS.",
            "aspect": "general",
            "chunk_ids": ["chunk_1"],
        }
    ]
    question_trees = {
        "general": _build_tree(
            "What is Apaleo's product?",
            "Apaleo is a hotel PMS.",
        )
    }
    new_chunks = [
        p.Chunk(
            chunk_id="chunk_new_1",
            text="Apaleo product is cloud-native hotel PMS with API-first architecture.",
            source_file="x.md",
            page_or_slide="1",
        )
    ]
    company = p.Company(name="Apaleo", industry="Hospitality software", tagline="", about="")

    result = asyncio.run(
        ea.enrich_qa_pairs_with_new_evidence(
            previous_qa_pairs=previous_qa_pairs,
            question_trees=question_trees,
            new_chunks=new_chunks,
            company=company,
        )
    )

    assert len(result) == 1
    # The LLM raised, so the previous answer must be preserved verbatim.
    assert result[0]["answer"] == "Apaleo is a hotel PMS."


def test_write_qa_pairs_into_trees_populates_node_answers(patched_enrichment):
    p = patched_enrichment
    ea = p.ea

    question_trees = {
        "general": _build_tree("Q1", ""),
        "team": _build_tree("Q2", ""),
    }
    qa_pairs = [
        {"question": "Q1", "answer": "A1", "chunk_ids": ["c1"]},
        {"question": "Q2", "answer": "A2", "chunk_ids": ["c2"]},
    ]
    ea.write_qa_pairs_into_trees(question_trees, qa_pairs)

    assert question_trees["general"].root_node.answer == "A1"
    assert question_trees["team"].root_node.answer == "A2"
    assert question_trees["general"].root_node.provenance["chunk_ids"] == ["c1"]
