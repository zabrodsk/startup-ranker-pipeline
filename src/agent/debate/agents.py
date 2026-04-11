"""Startup Agent and VC Agent for the structured debate.

Startup Agent: defends the company using ONLY retrieved evidence chunks (TF-IDF).
VC Agent: challenges the investment case using the VC's thesis + analysis summary.

Neither agent fabricates — both are grounded in their respective sources.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agent.ingest.store import Chunk, EvidenceStore
from agent.llm import create_llm
from agent.retrieval import retrieve_chunks

_logger = logging.getLogger(__name__)

DEBATE_CHUNK_K = 6        # chunks retrieved per debate turn
EXCERPT_CHARS = 200       # max chars per citation excerpt

# ---------------------------------------------------------------------------
# Startup Agent
# ---------------------------------------------------------------------------

STARTUP_SYSTEM_PROMPT = """\
You are the Startup Agent in an investment debate. Your role is to DEFEND the company
against the VC's concerns.

STRICT RULES:
- Answer ONLY using the evidence chunks provided below.
- Cite each claim with [chunk_id], e.g. [chunk_12].
- Do NOT fabricate facts, numbers, or claims not present in the evidence.
- If the evidence does not address a concern, say so honestly — do not invent an answer.
- Be concise: 3–5 sentences maximum per turn.
- Focus on directly addressing the VC's most recent argument.
"""

STARTUP_GAP_NOTE_TEMPLATE = """\

IMPORTANT — UNAVAILABLE EVIDENCE:
The founder has explicitly stated they cannot provide evidence for: {gap_note}
Do NOT invent data to fill this gap. Acknowledge the gap briefly, then pivot to
what the existing evidence CAN defend about the investment case.
"""

STARTUP_USER_PROMPT = """\
Company: {company_name}

VC's argument this round:
{vc_argument}

Relevant evidence chunks:
{chunks_text}

Defend the company using only the evidence above. Cite chunk IDs.
"""


async def startup_agent_turn(
    *,
    company_name: str,
    vc_argument: str,
    store: EvidenceStore,
    gap_note: str | None = None,
) -> dict[str, Any]:
    """Generate one Startup Agent turn.

    Retrieves relevant chunks via TF-IDF, then grounds the response in them.
    Returns {content, citations}.

    If ``gap_note`` is provided, a system-prompt addendum instructs the agent
    to acknowledge that the founder explicitly declined to provide evidence
    for that topic. Used on the first startup turn after the founder responds
    ``unavailable`` to a paused debate.
    """
    chunks = retrieve_chunks(vc_argument, store, k=DEBATE_CHUNK_K)

    chunks_text = _format_chunks(chunks)

    system_prompt = STARTUP_SYSTEM_PROMPT
    if gap_note:
        system_prompt = system_prompt + STARTUP_GAP_NOTE_TEMPLATE.format(gap_note=gap_note)

    llm = create_llm(temperature=0.3)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=STARTUP_USER_PROMPT.format(
            company_name=company_name,
            vc_argument=vc_argument,
            chunks_text=chunks_text,
        )),
    ]

    response = await llm.ainvoke(messages)
    content = _coerce_text(response.content)

    citations = [
        {
            "chunk_id": c.chunk_id,
            "excerpt": c.text[:EXCERPT_CHARS],
            "source_file": c.source_file,
        }
        for c in chunks
    ]

    return {"content": content, "citations": citations}


# ---------------------------------------------------------------------------
# VC Agent
# ---------------------------------------------------------------------------

VCAction = Literal["argue", "request_evidence", "conclude"]


VC_SYSTEM_PROMPT = """\
You are the VC Agent in an investment debate. Your role is to CHALLENGE the investment case
from the perspective of a specific VC firm.

STRICT RULES:
- Base your arguments on the investment thesis and analysis summary provided.
- Identify specific weaknesses: missing metrics, stage mismatch, market concerns, team gaps.
- Be precise and evidence-specific — reference what the analysis found (or failed to find).
- Do NOT make up data. If you don't know something, say the evidence is insufficient.
- Be concise: 3–5 sentences per turn.
- Escalate pressure each round — earlier rounds raise concerns, later rounds demand specifics.

EVERY RESPONSE MUST BE VALID JSON in this exact shape:

{{
  "action": "argue" | "request_evidence" | "conclude",
  "content": "<your message text>",
  "info_request": {{
    "topic": "<short label, e.g. 'Customer retention data'>",
    "questions": ["<specific question 1>", "<specific question 2>"],
    "rationale": "<why this evidence matters for the investment decision>"
  }} | null
}}

ACTION CHOICE RULES:
- Use "argue" by default: continue the debate with your next challenge.
- Use "request_evidence" ONLY when a specific factual claim critical to your
  decision cannot be verified from the analysis summary AND the founder could
  plausibly provide it (e.g. updated churn numbers, a new customer list, a
  missing team bio). When you choose this action, the debate will PAUSE and
  the founder will be prompted to upload new evidence or decline. Make your
  questions concrete and answerable. Populate "info_request".
- Use "conclude" ONLY when you genuinely have no further meaningful questions
  — either because the investment case is clear in either direction, or
  because asking more would be unproductive. When you choose this action the
  debate is FINALISED immediately. Leave "info_request" as null.

If in doubt, prefer "argue". Set "info_request" to null unless action is
"request_evidence".
"""

VC_USER_PROMPT = """\
VC Investment Thesis:
{vc_thesis}

Analysis Summary for {company_name}:
- Strategy Fit: {strategy_fit}/100 — {strategy_fit_summary}
- Team: {team_score}/100 — {team_summary}
- Potential: {potential_score}/100 — {potential_summary}
- Key red flags: {red_flags}

Startup's previous argument:
{startup_argument}

Round {current_round} of {max_rounds}.

Challenge the startup's argument. Focus on the most critical unresolved concern.
Respond with the JSON schema described in the system prompt.
"""


def _parse_vc_response(raw: str) -> dict[str, Any]:
    """Parse the VC agent's JSON response with a safe fallback.

    If the model emits a fenced code block or a JSON object embedded in prose,
    we still try to extract it. On any parse failure we return an ``argue``
    fallback so the debate loop never hangs on malformed output.
    """
    if not raw:
        return {"action": "argue", "content": "", "info_request": None}

    candidate = raw.strip()

    # Strip ``` fences if the model wrapped its JSON
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        _logger.warning(
            "VC agent returned non-JSON output; falling back to argue. error=%s", exc
        )
        return {"action": "argue", "content": raw.strip(), "info_request": None}

    if not isinstance(parsed, dict):
        return {"action": "argue", "content": raw.strip(), "info_request": None}

    action = parsed.get("action")
    if action not in ("argue", "request_evidence", "conclude"):
        _logger.warning("VC agent emitted unknown action=%r; coercing to argue", action)
        action = "argue"

    content = parsed.get("content")
    if not isinstance(content, str) or not content.strip():
        content = raw.strip()

    info_request = parsed.get("info_request")
    if action != "request_evidence":
        info_request = None
    elif not isinstance(info_request, dict):
        # Malformed request_evidence payload — downgrade to argue so we don't
        # leave the founder stranded with an empty request.
        _logger.warning(
            "VC agent emitted request_evidence without info_request; downgrading to argue"
        )
        action = "argue"
        info_request = None
    else:
        # Normalise shape
        topic = str(info_request.get("topic") or "").strip() or "Additional evidence"
        rationale = str(info_request.get("rationale") or "").strip()
        raw_questions = info_request.get("questions") or []
        if isinstance(raw_questions, str):
            raw_questions = [raw_questions]
        questions = [str(q).strip() for q in raw_questions if str(q).strip()]
        if not questions:
            _logger.warning(
                "VC agent emitted request_evidence with no questions; downgrading to argue"
            )
            action = "argue"
            info_request = None
        else:
            info_request = {
                "topic": topic,
                "questions": questions,
                "rationale": rationale,
            }

    return {"action": action, "content": content.strip(), "info_request": info_request}


async def vc_agent_turn(
    *,
    company_name: str,
    vc_thesis: str,
    analysis_summary: dict[str, Any],
    startup_argument: str,
    current_round: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Generate one VC Agent turn.

    Uses the VC's thesis and the analysis summary to challenge the startup.
    Returns ``{action, content, citations, info_request}`` where:
      - ``action`` ∈ {"argue", "request_evidence", "conclude"}
      - ``citations`` is always empty for the VC agent
      - ``info_request`` is populated only when ``action == 'request_evidence'``
    """
    ranking = analysis_summary or {}
    red_flags = ", ".join(ranking.get("red_flags") or []) or "none identified"

    llm = create_llm(temperature=0.4)
    messages = [
        SystemMessage(content=VC_SYSTEM_PROMPT),
        HumanMessage(content=VC_USER_PROMPT.format(
            vc_thesis=vc_thesis or "(no specific thesis provided)",
            company_name=company_name,
            strategy_fit=_fmt_score(ranking.get("strategy_fit_score")),
            strategy_fit_summary=ranking.get("strategy_fit_summary") or "—",
            team_score=_fmt_score(ranking.get("team_score")),
            team_summary=ranking.get("team_summary") or "—",
            potential_score=_fmt_score(ranking.get("upside_score")),
            potential_summary=ranking.get("potential_summary") or "—",
            red_flags=red_flags,
            startup_argument=startup_argument or "(no previous argument)",
            current_round=current_round,
            max_rounds=max_rounds,
        )),
    ]

    response = await llm.ainvoke(messages)
    raw_content = _coerce_text(response.content)
    parsed = _parse_vc_response(raw_content)

    return {
        "action": parsed["action"],
        "content": parsed["content"],
        "citations": [],
        "info_request": parsed["info_request"],
    }


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """\
You are a neutral investment analyst summarising a structured debate between
a VC and a startup. Be balanced, concise, and factual.
"""

SUMMARY_USER_PROMPT = """\
Company: {company_name}

The following debate took place over {num_rounds} rounds. Summarise:
1. The VC's main concerns (2–3 bullet points)
2. The startup's strongest defences (2–3 bullet points)
3. Key unresolved questions (1–2 bullet points)
4. Overall verdict: is this investment thesis strengthened or weakened by the debate?

Debate transcript:
{transcript}
"""


async def generate_debate_summary(
    *,
    company_name: str,
    messages: list[dict[str, Any]],
    num_rounds: int,
) -> str:
    """Generate a neutral summary of a completed debate."""
    transcript_lines = []
    for m in messages:
        # system_note rows are audit-only bookkeeping (re-eval ran, VC concluded,
        # etc.) and would poison the summary if left in the transcript.
        if m.get("message_type") == "system_note":
            continue
        speaker = m.get("speaker", "unknown").replace("_", " ").title()
        content = m.get("content", "")
        msg_type = m.get("message_type")
        if msg_type == "evidence_request":
            info = m.get("info_request") or {}
            topic = info.get("topic") if isinstance(info, dict) else None
            label = f"{speaker} (requested evidence: {topic})" if topic else f"{speaker} (requested evidence)"
        elif msg_type == "founder_response":
            response_type = m.get("founder_response_type") or "acknowledged"
            label = f"Founder ({response_type})"
        else:
            label = speaker
        transcript_lines.append(f"[Round {m.get('round', '?')} — {label}]\n{content}")

    transcript = "\n\n".join(transcript_lines)

    llm = create_llm(temperature=0.2)
    messages_llm = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=SUMMARY_USER_PROMPT.format(
            company_name=company_name,
            num_rounds=num_rounds,
            transcript=transcript[:8000],  # cap to avoid token overflow
        )),
    ]

    response = await llm.ainvoke(messages_llm)
    return _coerce_text(response.content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "No relevant evidence found."
    parts = []
    for c in chunks:
        parts.append(f"[{c.chunk_id}] (source: {c.source_file})\n{c.text[:400]}")
    return "\n\n".join(parts)


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(
            (item.get("text") if isinstance(item, dict) else str(item)) or ""
            for item in value
        ).strip()
    return str(value)


def _fmt_score(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return str(round(float(value), 1))
    except Exception:
        return str(value)
