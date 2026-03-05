"""Section synthesis for person intelligence output."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.llm import create_llm
from agent.person_intel.extract import enforce_safe_risk_text
from agent.person_intel.models import ExtractedFact
from agent.pipeline.state.schemas import PersonProfileSections

_SYSTEM_PROMPT = """You generate a structured person profile from evidence-grounded facts.
Rules:
- Use only facts provided by the caller.
- Tone: professional, human, crisp, and readable by investors.
- Prefer concrete facts, outcomes, and role transitions over generic wording.
- If evidence is weak, phrase uncertainty explicitly rather than inventing facts.
- Do not infer private data.
- Keep language neutral and non-defamatory.
- Strength bullets must start with ✅.
- Section goals:
  - more_details: coherent narrative paragraph of career trajectory and scope.
  - biggest_achievements: outcome-oriented bullets with scale/time/context when present.
  - values_beliefs: inferred from public actions/statements, no mind-reading.
  - top_risk: execution-focused uncertainty, never defamatory.
"""


class _RawPersonProfileSections(BaseModel):
    interests_lifestyle: str
    strengths: list[str]
    more_details: str
    biggest_achievements: list[str]
    values_beliefs: str
    key_points: list[str]
    coolest_fact: str
    top_risk: str


def _normalize_sentences(text: str, min_sentences: int, max_sentences: int, filler: str) -> str:
    raw_parts = [p.strip() for p in (text or "").replace("\n", " ").split(".") if p.strip()]
    if not raw_parts:
        raw_parts = [filler.strip()]

    while len(raw_parts) < min_sentences:
        raw_parts.append(filler.strip())

    normalized = raw_parts[:max_sentences]
    return " ".join(f"{part.rstrip('.')}." for part in normalized if part)


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = " ".join((v or "").split()).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _repair_sections(raw: _RawPersonProfileSections) -> PersonProfileSections:
    strengths = _dedupe_ordered(raw.strengths or [])
    strengths = [s if s.startswith("✅") else f"✅ {s}" for s in strengths][:8]
    if len(strengths) < 5:
        strengths.extend(
            [
                "✅ Demonstrates consistent execution across prior roles.",
                "✅ Translates strategic thinking into operational action.",
                "✅ Builds credibility through evidence-backed communication.",
                "✅ Shows adaptability across changing business contexts.",
                "✅ Maintains disciplined focus on measurable outcomes.",
            ][: 5 - len(strengths)]
        )

    achievements = _dedupe_ordered(raw.biggest_achievements or [])[:7]
    if len(achievements) < 3:
        achievements.extend(
            [
                "Publicly documented achievements remain limited in available sources.",
                "Evidence points to professional progression, with limited quantified milestones.",
                "Independent third-party coverage of specific outcomes is still sparse.",
            ][: 3 - len(achievements)]
        )

    key_points = _dedupe_ordered(raw.key_points or [])[:8]
    if len(key_points) < 5:
        source_pool = _dedupe_ordered((raw.strengths or []) + (raw.biggest_achievements or []))
        for item in source_pool:
            if len(key_points) >= 8:
                break
            if item not in key_points:
                key_points.append(item)
    if len(key_points) < 5:
        key_points.extend(
            [
                "Public evidence confirms professional leadership activity.",
                "Recent role trajectory indicates active execution responsibility.",
                "Available sources show exposure to strategy and operations.",
                "External references suggest focus on measurable outcomes.",
                "Some profile areas remain weakly documented in public data.",
            ][: 5 - len(key_points)]
        )

    interests = _normalize_sentences(
        raw.interests_lifestyle or "",
        min_sentences=2,
        max_sentences=5,
        filler="Public evidence on interests and lifestyle remains limited.",
    )
    values = _normalize_sentences(
        raw.values_beliefs or "",
        min_sentences=2,
        max_sentences=5,
        filler="Values and beliefs are inferred cautiously from public actions.",
    )
    cool = _normalize_sentences(
        raw.coolest_fact or "",
        min_sentences=1,
        max_sentences=2,
        filler="A notable insight is the combination of strategy experience and public-facing execution.",
    )
    top_risk = enforce_safe_risk_text((raw.top_risk or "").strip() or "Execution risk remains the key uncertainty due to incomplete public evidence.")
    details = " ".join((raw.more_details or "").split()).strip()
    if not details:
        details = "Public evidence indicates meaningful professional experience, though parts of the career timeline remain only partially documented."

    return PersonProfileSections(
        interests_lifestyle=interests,
        strengths=strengths,
        more_details=details,
        biggest_achievements=achievements,
        values_beliefs=values,
        key_points=key_points,
        coolest_fact=cool,
        top_risk=top_risk,
    )


def _fallback_sections(facts: list[ExtractedFact], unknowns: list[str]) -> PersonProfileSections:
    by_section: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        by_section[fact.section].append(fact.text)

    interests = by_section["interests_lifestyle"][:2]
    interests_text = (
        ". ".join(interests) + "."
        if interests
        else "Unknown lifestyle details from available public evidence. Interests are not explicitly documented."
    )
    interests_text = _normalize_sentences(
        interests_text,
        min_sentences=2,
        max_sentences=5,
        filler="Public evidence on lifestyle and interests remains limited.",
    )

    strengths_texts = by_section["strengths"][:8]
    if len(strengths_texts) < 5:
        strengths_texts.extend(by_section["key_points"][: 5 - len(strengths_texts)])
    if len(strengths_texts) < 5:
        strengths_texts.extend(
            [
                "✅ Public signals indicate practical leadership experience.",
                "✅ Evidence suggests a structured approach to execution.",
                "✅ Demonstrates ability to operate across complex contexts.",
                "✅ Communicates in a fact-oriented professional style.",
                "✅ Maintains continuity across strategic and operational roles.",
            ]
        )
    strengths = [s if s.startswith("✅") else f"✅ {s}" for s in strengths_texts[:8]]

    details = by_section["more_details"][:3]
    more_details = (
        " ".join(details)
        if details
        else "Public evidence is limited, so detailed biography and chronology remain unknown."
    )

    achievements = by_section["biggest_achievements"][:7]
    if len(achievements) < 3:
        achievements.extend(
            [
                "Publicly documented milestone detail is limited in available sources.",
                "Professional trajectory is visible, but outcome metrics are incomplete.",
                "Additional independent references would strengthen achievement verification.",
            ]
        )
    achievements = achievements[:7]

    values = by_section["values_beliefs"][:2]
    values_text = (
        ". ".join(values) + "."
        if values
        else "Values and beliefs are mostly unknown from current sources. Available signals suggest professional focus, but explicit principles are sparse."
    )
    values_text = _normalize_sentences(
        values_text,
        min_sentences=2,
        max_sentences=5,
        filler="Values and beliefs are weakly evidenced in public sources.",
    )

    key_points = by_section["key_points"][:8]
    if len(key_points) < 5:
        for extra in by_section["strengths"] + by_section["more_details"] + by_section["biggest_achievements"]:
            if len(key_points) >= 8:
                break
            if extra and extra not in key_points:
                key_points.append(extra)
    if len(key_points) < 5:
        key_points.extend(
            [
                "Current public evidence provides partial but usable profile coverage.",
                "Leadership and execution themes appear repeatedly across sources.",
                "Public data quality varies by topic and timeframe.",
                "Some claims are well supported while others remain thinly documented.",
                "Further source depth would improve confidence in edge-case claims.",
            ]
        )
    key_points = key_points[:8]

    cool = by_section["coolest_fact"][:1]
    coolest_fact = (
        cool[0] + "."
        if cool
        else "The most notable insight is how much can be established while preserving strict evidence-only constraints."
    )

    risk_candidates = by_section["top_risk"][:1]
    if risk_candidates:
        top_risk = risk_candidates[0]
    elif unknowns:
        top_risk = unknowns[0]
    else:
        top_risk = "The main risk is uncertainty from incomplete public evidence coverage."

    top_risk = enforce_safe_risk_text(top_risk)

    return _repair_sections(
        _RawPersonProfileSections(
        interests_lifestyle=interests_text,
        strengths=strengths,
        more_details=more_details,
        biggest_achievements=achievements,
        values_beliefs=values_text,
        key_points=key_points,
        coolest_fact=coolest_fact,
        top_risk=top_risk,
        )
    )


async def synthesize_sections(
    facts: list[ExtractedFact],
    unknowns: list[str],
) -> PersonProfileSections:
    """Synthesize final sections from facts, with deterministic fallback."""
    fallback = _fallback_sections(facts, unknowns)

    fact_lines = []
    for fact in facts:
        fact_lines.append(
            f"- section={fact.section} confidence={fact.confidence} status={fact.status} text={fact.text}"
        )

    prompt = (
        "Facts:\n"
        + "\n".join(fact_lines[:120])
        + "\n\nUnknowns:\n"
        + "\n".join(f"- {u}" for u in unknowns[:20])
        + "\n\nReturn valid structured output."
    )

    try:
        llm = create_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(_RawPersonProfileSections)
        output: _RawPersonProfileSections = await structured_llm.ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return _repair_sections(output)
    except Exception:
        return fallback
