"""Tests for decomposition route tagging + the planner hybrid-prompt variant (PR4)."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import evidence_answering as ea
from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode
from agent.ingest.store import Chunk
from agent.pipeline.stages import cache as tree_cache
from agent.pipeline.stages.decomposition import (
    _build_question_tree_from_decomposition_tree,
)
from agent.pipeline.state.decomposition import DecompositionNode, DecompositionTree
from agent.prompt_library import manager as prompt_manager
from agent.prompt_library.defaults import (
    EVIDENCE_HYBRID_SYSTEM_PROMPT,
    EVIDENCE_HYBRID_WEB_PLANNER_SYSTEM_PROMPT,
    ORDERED_PROMPT_IDS,
)
from agent.prompt_library.manager import get_prompt


# --- Decomposition schema -----------------------------------------------------


def test_decomposition_node_parses_old_payload_without_route():
    node = DecompositionNode(**{"question": "Q?", "sub_questions": []})
    assert node.route is None


def test_decomposition_node_parses_route():
    node = DecompositionNode(question="Q?", sub_questions=[], route="sector_market")
    assert node.route == "sector_market"


def test_question_node_parses_old_payload_without_route():
    node = QuestionNode(**{"question": "Q?", "answer": "A", "sub_nodes": []})
    assert node.route is None


def test_tree_builder_propagates_normalized_routes():
    tree = DecompositionTree(
        nodes=[
            DecompositionNode(question="Root?", sub_questions=["A?", "B?"], route="Sector-Market"),
            DecompositionNode(question="A?", sub_questions=[], route="competitors"),
            DecompositionNode(question="B?", sub_questions=[], route="not_a_route"),
        ]
    )
    qt = _build_question_tree_from_decomposition_tree(tree, "market")
    assert qt.root_node.route == "sector_market"  # normalized from "Sector-Market"
    by_q = {n.question: n for n in qt.root_node.sub_nodes}
    assert by_q["A?"].route == "competitors"
    assert by_q["B?"].route is None  # unknown tag -> None -> fallback router


def test_tree_builder_orphan_children_get_no_route():
    tree = DecompositionTree(
        nodes=[
            DecompositionNode(question="Root?", sub_questions=["Orphan?"], route="regulation"),
        ]
    )
    qt = _build_question_tree_from_decomposition_tree(tree, "market")
    assert qt.root_node.sub_nodes[0].route is None


def test_route_tagging_instruction_lists_all_routes():
    from agent.web_search.planner import ROUTE_TAGGING_INSTRUCTION, QuestionRoute

    for route in QuestionRoute:
        assert f'"{route.value}"' in ROUTE_TAGGING_INSTRUCTION


def test_decomposition_cache_name_bumped_for_route_tags():
    assert tree_cache.CACHE_NAME == "question_trees_v2.json"


# --- Prompt registry -----------------------------------------------------------


def test_web_planner_prompt_id_registered_with_default():
    assert "evidence.hybrid.system.web_planner" in ORDERED_PROMPT_IDS
    value = get_prompt("evidence.hybrid.system.web_planner")
    assert value == EVIDENCE_HYBRID_WEB_PLANNER_SYSTEM_PROMPT
    assert "LEGITIMATE primary source" in value
    assert "ONLY a fallback" not in value


def test_old_persisted_catalog_without_new_key_falls_back_to_default(monkeypatch, tmp_path):
    # Simulate a stale persisted library.json that predates the new prompt id.
    legacy = {"evidence.hybrid.system": EVIDENCE_HYBRID_SYSTEM_PROMPT}
    library_path = tmp_path / "library.json"
    library_path.write_text(json.dumps(legacy))
    monkeypatch.setattr(prompt_manager, "LIBRARY_PATH", library_path)

    assert get_prompt("evidence.hybrid.system") == EVIDENCE_HYBRID_SYSTEM_PROMPT
    assert (
        get_prompt("evidence.hybrid.system.web_planner")
        == EVIDENCE_HYBRID_WEB_PLANNER_SYSTEM_PROMPT
    )


# --- Call-site prompt selection by flag -----------------------------------------

THIN_ANSWER = "Insufficient information available."
SECTOR_RESULT = (
    "The European market for wildfire detection sensors is projected to grow "
    "rapidly, with adoption by utilities and forest agencies and rising demand "
    "for early-warning monitoring systems across the region."
)


class RecordingLLM:
    def __init__(self, sink):
        self.sink = sink

    async def ainvoke(self, messages):
        self.sink.append(messages)
        if "Web Search Results" in messages[-1].content:
            return SimpleNamespace(content="Hybrid answer [web].")
        return SimpleNamespace(content=THIN_ANSWER)


@pytest.fixture()
def harness(monkeypatch):
    calls: list = []
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: RecordingLLM(calls))
    monkeypatch.setattr(
        ea, "retrieve_chunks",
        lambda question, store, k=8: [
            Chunk(chunk_id="chunk_0", text="Deck text.", source_file="deck", page_or_slide="1")
        ],
    )
    monkeypatch.setattr(
        ea, "_run_web_search",
        lambda *args, **kwargs: SECTOR_RESULT,
    )
    monkeypatch.delenv("RDI_WEB_EVIDENCE_PLANNER", raising=False)
    return calls


def _run(question: str, calls):
    company = Company(
        name="Mantic", industry="wildfire detection sensors",
        domain="mantic.ai", geo="Europe",
    )
    state = {"count": [0], "lock": asyncio.Lock(), "max": 10}
    asyncio.run(
        ea.answer_question_from_evidence(
            question, company, store=None, use_web_search=True, web_search_state=state
        )
    )
    return [m[0].content for m in calls if "Web Search Results" in m[-1].content]


def test_on_mode_hybrid_uses_web_planner_prompt(harness, monkeypatch):
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    hybrid_system = _run(
        "Is the sector attractive given market size, growth and regulation?", harness
    )
    assert len(hybrid_system) == 1
    assert "LEGITIMATE primary source" in hybrid_system[0]
    assert "ONLY a fallback" not in hybrid_system[0]


@pytest.mark.parametrize("mode", [None, "off", "shadow"])
def test_off_and_shadow_modes_keep_legacy_hybrid_prompt(harness, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("RDI_WEB_EVIDENCE_PLANNER", raising=False)
    else:
        monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", mode)
    # Use a question whose legacy gate accepts the sector text (company name
    # is irrelevant here: pick a question matching the result tokens AND
    # make the company name appear in results to satisfy the legacy gate).
    monkeypatch.setattr(
        ea, "_run_web_search",
        lambda *args, **kwargs: "Mantic " + SECTOR_RESULT,
    )
    hybrid_system = _run(
        "What is the market size and adoption of wildfire detection sensors in Europe?",
        harness,
    )
    assert len(hybrid_system) == 1
    assert hybrid_system[0] == EVIDENCE_HYBRID_SYSTEM_PROMPT
