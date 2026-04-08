"""Debate orchestrator.

Runs a structured multi-round debate between the VC Agent and Startup Agent.
Persists every message to Supabase and supports pause/resume.

Round structure (default 3 rounds):
  Round N:
    1. VC Agent raises concerns (or opens with the analysis red flags on round 1)
    2. Startup Agent responds with evidence-grounded defence

After all rounds: generate summary, mark debate completed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from agent.debate.agents import (
    generate_debate_summary,
    startup_agent_turn,
    vc_agent_turn,
)
from agent.ingest.store import Chunk, EvidenceStore

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3

# Opening prompt sent by the VC agent on round 1 when there's no prior startup argument
_OPENING_PROMPT = (
    "Please open by presenting your strongest investment case. "
    "The analysis has identified some concerns — I will challenge them."
)


def _build_store_from_chunks(company_id: str, chunks: list[dict[str, Any]]) -> EvidenceStore:
    store = EvidenceStore(startup_slug=company_id)
    for c in chunks:
        store.chunks.append(
            Chunk(
                chunk_id=str(c.get("chunk_id") or c.get("id") or ""),
                text=c.get("text") or "",
                source_file=c.get("source_file") or "",
                page_or_slide=c.get("page_or_slide") or 0,
            )
        )
    return store


async def run_debate(
    *,
    debate_id: str,
    company_name: str,
    vc_thesis: str,
    analysis_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    existing_messages: list[dict[str, Any]],
    current_round: int,
    max_rounds: int,
    db_module: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Run (or resume) a debate from current_round to max_rounds.

    Yields each message dict as it is generated so the caller can stream
    updates to connected WebSocket clients.

    Persists every message to Supabase via db_module.
    """
    store = _build_store_from_chunks(company_name, chunks)

    # Find last startup argument for VC agent context
    last_startup_msg = _last_speaker_content(existing_messages, "startup_agent")
    last_vc_msg = _last_speaker_content(existing_messages, "vc_agent")

    for round_num in range(current_round, max_rounds + 1):
        logger.info("Debate %s — starting round %d/%d", debate_id, round_num, max_rounds)

        # --- VC Agent turn ---
        vc_input = last_startup_msg if last_startup_msg else _OPENING_PROMPT
        try:
            vc_result = await vc_agent_turn(
                company_name=company_name,
                vc_thesis=vc_thesis,
                analysis_summary=analysis_summary,
                startup_argument=vc_input,
                current_round=round_num,
                max_rounds=max_rounds,
            )
        except Exception as exc:
            logger.error("VC agent failed on round %d: %s", round_num, exc)
            break

        vc_msg = db_module.save_debate_message(
            debate_id=debate_id,
            round=round_num,
            speaker="vc_agent",
            content=vc_result["content"],
            citations=vc_result.get("citations") or [],
        )
        last_vc_msg = vc_result["content"]
        yield vc_msg or {
            "debate_id": debate_id,
            "round": round_num,
            "speaker": "vc_agent",
            "content": vc_result["content"],
            "citations": vc_result.get("citations") or [],
        }

        # Update round in DB
        db_module.update_debate_round(debate_id, round_num)

        # --- Startup Agent turn ---
        try:
            startup_result = await startup_agent_turn(
                company_name=company_name,
                vc_argument=last_vc_msg,
                store=store,
            )
        except Exception as exc:
            logger.error("Startup agent failed on round %d: %s", round_num, exc)
            break

        startup_msg = db_module.save_debate_message(
            debate_id=debate_id,
            round=round_num,
            speaker="startup_agent",
            content=startup_result["content"],
            citations=startup_result.get("citations") or [],
        )
        last_startup_msg = startup_result["content"]
        yield startup_msg or {
            "debate_id": debate_id,
            "round": round_num,
            "speaker": "startup_agent",
            "content": startup_result["content"],
            "citations": startup_result.get("citations") or [],
        }

    # --- Generate summary ---
    logger.info("Debate %s — generating summary", debate_id)
    all_messages = db_module.get_debate_messages(debate_id)
    try:
        summary = await generate_debate_summary(
            company_name=company_name,
            messages=all_messages,
            num_rounds=max_rounds,
        )
    except Exception as exc:
        logger.error("Summary generation failed: %s", exc)
        summary = "Summary could not be generated."

    db_module.complete_debate(debate_id, summary)

    yield {
        "debate_id": debate_id,
        "round": max_rounds,
        "speaker": "system",
        "content": summary,
        "citations": [],
        "is_summary": True,
    }


def _last_speaker_content(messages: list[dict[str, Any]], speaker: str) -> str:
    for m in reversed(messages):
        if m.get("speaker") == speaker:
            return m.get("content") or ""
    return ""
