"""Debate orchestrator.

Runs a structured multi-round debate between the VC Agent and Startup Agent.
Persists every message to Supabase and supports pause/resume.

Round structure (default 3 rounds):
  Round N:
    1. VC Agent speaks. One of three outcomes:
       - action='argue': normal challenge → persist + yield, continue to startup turn.
       - action='request_evidence': the VC needs more data → persist the evidence
         request, pause the debate awaiting founder input, raise
         ``DebatePausedForEvidence`` to the caller so the background task shuts
         down cleanly.
       - action='conclude': the VC has no further questions → persist a
         ``system_note``, skip the startup turn, jump to summary + complete.
    2. Startup Agent responds with evidence-grounded defence.

After the final round (or an early 'conclude'): generate summary, mark debate
completed.
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


class DebatePausedForEvidence(Exception):
    """Raised by ``run_debate`` when the VC agent requested more evidence.

    The debate is already marked ``paused`` + ``awaiting_input_from='founder'``
    in the database by the time this is raised. The background task wrapper in
    ``web/app.py`` catches this sentinel and returns cleanly instead of logging
    the run as a failure.
    """


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


def _compute_resume_gap_note(existing_messages: list[dict[str, Any]]) -> str | None:
    """If the last meaningful response was ``founder_response='unavailable'``,
    derive the gap topic from the preceding ``evidence_request`` message.

    Returns the topic string to inject into the next startup agent turn, or
    ``None`` if no gap note applies.
    """
    if not existing_messages:
        return None
    # Walk backwards for the most recent founder_response
    last_founder_idx: int | None = None
    for idx in range(len(existing_messages) - 1, -1, -1):
        m = existing_messages[idx]
        if m.get("message_type") == "founder_response":
            last_founder_idx = idx
            break
    if last_founder_idx is None:
        return None
    last_founder = existing_messages[last_founder_idx]
    if last_founder.get("founder_response_type") != "unavailable":
        return None
    # Walk backwards from that point looking for the paired evidence_request
    for idx in range(last_founder_idx - 1, -1, -1):
        m = existing_messages[idx]
        if m.get("message_type") == "evidence_request":
            info = m.get("info_request")
            if isinstance(info, dict):
                topic = info.get("topic")
                if isinstance(topic, str) and topic.strip():
                    return topic.strip()
            return None
    return None


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
    context_notes: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run (or resume) a debate from current_round to max_rounds.

    Yields each message dict as it is generated so the caller can stream
    updates to connected WebSocket clients.

    Persists every message to Supabase via db_module.

    ``context_notes`` is an optional hint passed by the resume path. When it
    contains ``{"gap_note": "..."}`` (set by the founder-response endpoint on
    an ``unavailable`` response), the FIRST startup turn after resume will
    instruct the agent to acknowledge that the founder cannot provide that
    evidence and pivot to what is defensible. If the caller omits it, we fall
    back to deriving the gap note from ``existing_messages``.
    """
    store = _build_store_from_chunks(company_name, chunks)

    # Find last startup argument for VC agent context
    last_startup_msg = _last_meaningful_content(existing_messages, "startup_agent")
    last_vc_msg = _last_meaningful_content(existing_messages, "vc_agent")

    # Gap note: caller override first, then derive from history
    gap_note: str | None = None
    if context_notes and isinstance(context_notes.get("gap_note"), str):
        gap_note = context_notes["gap_note"].strip() or None
    if gap_note is None:
        gap_note = _compute_resume_gap_note(existing_messages)

    # The gap note only applies to the NEXT startup turn. Track whether it has
    # been consumed so subsequent turns in the same run don't re-apply it.
    gap_note_consumed = False

    # Did we finish the round loop by the VC concluding early? If so, skip
    # straight to summary + complete without an error.
    concluded_early = False

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

        vc_action = vc_result.get("action", "argue")

        # ---- VC chose to CONCLUDE ----
        if vc_action == "conclude":
            logger.info(
                "Debate %s — VC agent concluded on round %d/%d", debate_id, round_num, max_rounds
            )
            conclusion_note = (
                vc_result.get("content") or "VC agent concluded: no further questions."
            )
            note_msg = db_module.save_debate_message(
                debate_id=debate_id,
                round=round_num,
                speaker="system",
                content=f"VC agent concluded: {conclusion_note}",
                citations=[],
                message_type="system_note",
            )
            yield note_msg or {
                "debate_id": debate_id,
                "round": round_num,
                "speaker": "system",
                "content": f"VC agent concluded: {conclusion_note}",
                "citations": [],
                "message_type": "system_note",
            }
            db_module.update_debate_round(debate_id, round_num)
            concluded_early = True
            break

        # ---- VC chose to REQUEST EVIDENCE ----
        if vc_action == "request_evidence":
            info_request = vc_result.get("info_request") or {}
            logger.info(
                "Debate %s — VC agent requested evidence on round %d: topic=%r",
                debate_id,
                round_num,
                info_request.get("topic") if isinstance(info_request, dict) else None,
            )
            evidence_msg = db_module.save_debate_message(
                debate_id=debate_id,
                round=round_num,
                speaker="vc_agent",
                content=vc_result.get("content") or "",
                citations=[],
                message_type="evidence_request",
                info_request=info_request,
            )
            # Yield BEFORE pausing so any live WS subscriber sees the request.
            yield evidence_msg or {
                "debate_id": debate_id,
                "round": round_num,
                "speaker": "vc_agent",
                "content": vc_result.get("content") or "",
                "citations": [],
                "message_type": "evidence_request",
                "info_request": info_request,
            }
            db_module.update_debate_round(debate_id, round_num)
            topic = (
                info_request.get("topic")
                if isinstance(info_request, dict)
                else None
            )
            db_module.pause_debate_for_evidence(debate_id, topic=topic)
            raise DebatePausedForEvidence(debate_id)

        # ---- VC chose to ARGUE (default) ----
        vc_msg = db_module.save_debate_message(
            debate_id=debate_id,
            round=round_num,
            speaker="vc_agent",
            content=vc_result["content"],
            citations=vc_result.get("citations") or [],
            message_type="argument",
        )
        last_vc_msg = vc_result["content"]
        yield vc_msg or {
            "debate_id": debate_id,
            "round": round_num,
            "speaker": "vc_agent",
            "content": vc_result["content"],
            "citations": vc_result.get("citations") or [],
            "message_type": "argument",
        }

        # Update round in DB
        db_module.update_debate_round(debate_id, round_num)

        # --- Startup Agent turn ---
        turn_gap_note = None
        if not gap_note_consumed:
            turn_gap_note = gap_note
            gap_note_consumed = True
        try:
            startup_result = await startup_agent_turn(
                company_name=company_name,
                vc_argument=last_vc_msg,
                store=store,
                gap_note=turn_gap_note,
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
            message_type="argument",
        )
        last_startup_msg = startup_result["content"]
        yield startup_msg or {
            "debate_id": debate_id,
            "round": round_num,
            "speaker": "startup_agent",
            "content": startup_result["content"],
            "citations": startup_result.get("citations") or [],
            "message_type": "argument",
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
        "message_type": "system_note",
        "is_summary": True,
    }


def _last_meaningful_content(messages: list[dict[str, Any]], speaker: str) -> str:
    """Find the last actual-argument content for a speaker.

    Skips evidence_request, founder_response, and system_note messages — only
    ``message_type='argument'`` (or legacy rows with no ``message_type``) count
    as prior debate turns.
    """
    for m in reversed(messages):
        if m.get("speaker") != speaker:
            continue
        msg_type = m.get("message_type")
        if msg_type and msg_type != "argument":
            continue
        return m.get("content") or ""
    return ""
