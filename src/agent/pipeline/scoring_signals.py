"""Structured scoring signals for the ranking layer.

The ranking stage should not depend only on whatever the Q&A decomposition
happened to surface. This helper builds a compact, prompt-friendly signal pack
from Specter-derived evidence chunks and Q&A fallbacks.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from agent.dataclasses.company import Company
from agent.ingest.store import EvidenceStore


_TIER_1_INVESTORS = {
    "accel",
    "andreessen horowitz",
    "a16z",
    "benchmark",
    "dst global",
    "general catalyst",
    "gv",
    "greylock",
    "index ventures",
    "khosla",
    "khosla ventures",
    "kleiner perkins",
    "lightspeed",
    "openai",
    "sequoia",
    "sv angel",
    "y combinator",
}

_TIER_2_INVESTORS = {
    "8vc",
    "episode 1",
    "entrepreneurs first",
    "eurazeo",
    "headline",
    "htgf",
    "inovo",
    "inovo.vc",
    "menlo ventures",
    "paladin",
    "vertex",
}

_HIGHLIGHT_MAPPINGS = {
    "web traffic surge": "demand / market-pull signal",
    "headcount surge": "execution velocity and talent magnet",
    "product reviews scale-up": "willingness-to-pay / usage validation",
    "top tier investors": "investor validation",
    "recent funding": "momentum and capital access",
    "founder highlights": "founder archetype and Team evidence",
    "certifications": "enterprise readiness",
    "security": "enterprise readiness",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _append_unique(target: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in target:
        target.append(value)


def _parse_int(value: str) -> int | None:
    try:
        return int(value.replace(",", "").strip())
    except Exception:
        return None


def _classify_investor(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    if normalized in _TIER_1_INVESTORS:
        return "tier_1_top_global_or_elite_strategic"
    if normalized in _TIER_2_INVESTORS:
        return "tier_2_notable_institutional_or_specialist"
    return "tier_3_regional_angel_accelerator_or_unknown"


def _interest_recency(signal_date: str | None, *, now: datetime | None = None) -> str:
    if not signal_date:
        return "unknown"
    now = now or datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(signal_date.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0, (now - parsed).days)
    if age_days <= 183:
        return "full_weight_0_6_months"
    if age_days <= 366:
        return "partial_weight_6_12_months"
    return "low_weight_over_12_months"


def _infer_stage(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("pre-seed", "pre seed")):
        return "pre_seed"
    if "seed" in lowered:
        return "seed"
    if "series a" in lowered:
        return "series_a"
    if "series b" in lowered or "series c" in lowered or "series d" in lowered:
        return "growth"
    if "late" in lowered:
        return "late"
    return ""


def _extract_after_label(text: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _scan_text_for_signals(signals: dict[str, Any], text: str, source: str) -> None:
    if not text:
        return
    lowered = text.lower()

    stage = _infer_stage(text)
    if stage and not signals.get("stage"):
        signals["stage"] = stage
        signals["stage_context"] = f"{stage.replace('_', '-')}-stage signal from {source}"

    employee = _extract_after_label(text, "Employee Count")
    if employee and signals.get("employee_count") is None:
        signals["employee_count"] = _parse_int(employee.split()[0])

    department_mix = _extract_after_label(text, "Department mix (latest)")
    if department_mix:
        signals["department_mix"] = department_mix

    clients = _extract_after_label(text, "Reported Clients")
    if clients:
        for client in clients.split(","):
            _append_unique(signals["reported_clients"], client)

    investors = _extract_after_label(text, "Investors")
    if investors:
        for investor in investors.split(","):
            name = investor.strip()
            if name:
                tier = _classify_investor(name)
                signals["investors"].append({"name": name, "tier": tier})

    for match in re.finditer(
        r"\[(?P<date>[^\]]+)\]\s+score\s+(?P<score>[^:]+):\s*(?P<summary>.+)",
        text,
        flags=re.IGNORECASE,
    ):
        signals["investor_interest_signals"].append(
            {
                "date": match.group("date").strip(),
                "score": match.group("score").strip(),
                "summary": match.group("summary").strip(),
                "recency_weight": _interest_recency(match.group("date").strip()),
            }
        )

    highlights = _extract_after_label(text, "Highlights")
    if highlights:
        for highlight in highlights.split(","):
            label = highlight.strip()
            if not label:
                continue
            mapped = _HIGHLIGHT_MAPPINGS.get(label.lower())
            signals["specter_highlights"].append(
                {"label": label, "mapped_factor": mapped or "general supporting signal"}
            )

    founder_highlights = _extract_after_label(text, "Founder Highlights")
    if founder_highlights:
        for highlight in founder_highlights.split(","):
            _append_unique(signals["founder_highlights"], highlight)

    if "github" in lowered or "gitlab" in lowered or "open source" in lowered:
        _append_unique(signals["young_founder_signals"], "GitHub/open-source/project activity")
    if "hackathon" in lowered or "competition" in lowered:
        _append_unique(signals["young_founder_signals"], "hackathon/competition signal")
    if "followers" in lowered or "audience" in lowered:
        _append_unique(signals["young_founder_signals"], "social audience/community signal")
    if any(school in lowered for school in ("mit", "stanford", "harvard", "cambridge", "oxford", "princeton")):
        _append_unique(signals["young_founder_signals"], "top-tier school or technical background")
    if "side project" in lowered or "early entrepreneurship" in lowered:
        _append_unique(signals["young_founder_signals"], "side project / early entrepreneurship")
    if "failed startup" in lowered or "previous startup failed" in lowered:
        _append_unique(signals["founder_archetype_evidence"], "serial founder with prior failure/learnings")
    if "early employee" in lowered or "high-growth startup" in lowered or "vc-backed startup" in lowered:
        _append_unique(signals["founder_archetype_evidence"], "previous high-growth startup operator")
    if "acquired" in lowered or "exit" in lowered:
        _append_unique(signals["founder_archetype_evidence"], "serial founder with exit")
    if any(token in lowered for token in ("google", "openai", "microsoft", "linkedin", "meta", "apple", "amazon")):
        _append_unique(signals["founder_archetype_evidence"], "big-tech R&D/product alumni")

    if "certification" in lowered or "iso 27001" in lowered or "soc 2" in lowered or "gdpr" in lowered:
        _append_unique(signals["enterprise_readiness_signals"], "certifications/security/compliance")
    if "web traffic" in lowered:
        _append_unique(signals["market_pull_signals"], "web traffic signal")
    if "product reviews" in lowered or "g2:" in lowered or "trustpilot" in lowered:
        _append_unique(signals["market_pull_signals"], "product review / usage validation")


def build_scoring_signals(
    company: Company,
    *,
    evidence_store: EvidenceStore | None = None,
    all_qa_pairs: list[dict[str, Any]] | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact structured signal pack for ranking prompts."""
    signals: dict[str, Any] = {
        "company_name": company.name,
        "stage": "",
        "stage_context": "",
        "employee_count": None,
        "department_mix": "",
        "headcount_growth": [],
        "reported_clients": [],
        "investors": [],
        "investor_interest_signals": [],
        "specter_highlights": [],
        "founder_highlights": [],
        "young_founder_signals": [],
        "founder_archetype_evidence": [],
        "enterprise_readiness_signals": [],
        "market_pull_signals": [],
    }
    if existing:
        for key, value in existing.items():
            if value not in (None, "", [], {}):
                signals[key] = value

    for text in (company.get_company_summary(),):
        _scan_text_for_signals(signals, text, "company profile")

    if evidence_store:
        for chunk in evidence_store.chunks or []:
            text = _safe_text(chunk.text)
            source = _safe_text(chunk.page_or_slide or chunk.source_file or "evidence")
            _scan_text_for_signals(signals, text, source)
            if "employee growth" in text.lower():
                for line in text.splitlines():
                    if "employee growth" in line.lower():
                        _append_unique(signals["headcount_growth"], line.strip())

    for qa in all_qa_pairs or []:
        text = f"{qa.get('question', '')}\n{qa.get('answer', '')}\n{qa.get('chunks_preview', '')}"
        _scan_text_for_signals(signals, text, f"Q&A:{qa.get('aspect') or 'unknown'}")

    return signals


def format_scoring_signals(signals: dict[str, Any] | None) -> str:
    """Render scoring signals as compact text for LLM prompts."""
    if not signals:
        return "No structured scoring signals available."
    lines: list[str] = []
    for key in (
        "stage",
        "stage_context",
        "employee_count",
        "department_mix",
        "headcount_growth",
        "reported_clients",
        "investors",
        "investor_interest_signals",
        "specter_highlights",
        "founder_highlights",
        "young_founder_signals",
        "founder_archetype_evidence",
        "enterprise_readiness_signals",
        "market_pull_signals",
    ):
        value = signals.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "No structured scoring signals available."
