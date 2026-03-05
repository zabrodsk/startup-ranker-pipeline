"""Markdown renderer for person profile intelligence output."""

from __future__ import annotations

from agent.pipeline.state.schemas import PersonProfileOutput


def render_person_profile_markdown(profile: PersonProfileOutput) -> str:
    """Render deterministic markdown from structured profile JSON."""
    sections = profile.sections

    lines: list[str] = []
    lines.append("## INTERESTS & LIFESTYLE")
    lines.append(sections.interests_lifestyle)
    lines.append("")

    lines.append("## STRENGTHS")
    for item in sections.strengths:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## MORE DETAILS")
    lines.append(sections.more_details)
    lines.append("")

    lines.append("## BIGGEST ACHIEVEMENTS")
    for item in sections.biggest_achievements:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## VALUES & BELIEFS")
    lines.append(sections.values_beliefs)
    lines.append("")

    lines.append("## KEY POINTS")
    for item in sections.key_points:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## COOLEST FACT")
    lines.append(sections.coolest_fact)
    lines.append("")

    lines.append("## TOP RISK")
    lines.append(sections.top_risk)

    return "\n".join(lines).strip() + "\n"
