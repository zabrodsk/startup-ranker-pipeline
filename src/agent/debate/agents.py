"""Startup Agent and VC Agent for the structured debate.

Startup Agent: defends the company using ONLY retrieved evidence chunks (TF-IDF).
VC Agent: challenges the investment case using the VC's thesis + analysis summary.

Neither agent fabricates — both are grounded in their respective sources.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.ingest.store import Chunk, EvidenceStore
from agent.llm import create_llm
from agent.retrieval import retrieve_chunks

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
) -> dict[str, Any]:
    """Generate one Startup Agent turn.

    Retrieves relevant chunks via TF-IDF, then grounds the response in them.
    Returns {content, citations}.
    """
    chunks = retrieve_chunks(vc_argument, store, k=DEBATE_CHUNK_K)

    chunks_text = _format_chunks(chunks)

    llm = create_llm(temperature=0.3)
    messages = [
        SystemMessage(content=STARTUP_SYSTEM_PROMPT),
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
"""


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
    Returns {content, citations} (citations is always empty for VC agent).
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
    content = _coerce_text(response.content)

    return {"content": content, "citations": []}


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
        speaker = m.get("speaker", "unknown").replace("_", " ").title()
        content = m.get("content", "")
        transcript_lines.append(f"[Round {m.get('round', '?')} — {speaker}]\n{content}")

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
