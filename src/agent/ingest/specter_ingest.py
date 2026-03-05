"""Specter CSV ingestion — parse company + people exports into Company & EvidenceStore.

Reads the fixed-format Specter exports (company-signals CSV and people-signals CSV),
joins people to companies via specter_person_id, and produces per-company structured
evidence chunks optimized for TF-IDF retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from agent.dataclasses.company import Company
from agent.dataclasses.person import Education, Experience, Person
from agent.ingest.store import Chunk, EvidenceStore


def _safe(val: Any) -> str | None:
    """Return None for NaN/empty, else stripped string. Handles lists from Excel/CSV."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, list):
        s = " ".join(str(x) for x in val).strip() if val else ""
        return s if s else None
    s = str(val).strip()
    return s if s else None


def _normalize_public_url(val: str | None) -> str | None:
    """Normalize URL-like strings to explicit https:// when missing a scheme."""
    if not val:
        return None
    v = val.strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    if v.startswith("www.") or "linkedin.com/" in v:
        return f"https://{v.lstrip('/')}"
    return v


def _safe_float(val: Any) -> float | None:
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _parse_json_field(val: Any) -> list | dict | None:
    if pd.isna(val) or not val:
        return None
    s = str(val).strip()
    if not s or s in ("[]", "{}"):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# People parsing
# ---------------------------------------------------------------------------

def _parse_person(row: pd.Series) -> Person:
    """Convert a Specter people CSV row into a Person object."""
    education_list: list[Education] = []
    raw_edu = _parse_json_field(row.get("Education"))
    if isinstance(raw_edu, list):
        for e in raw_edu:
            if not isinstance(e, dict):
                continue
            education_list.append(Education(
                institution=_safe(e.get("Name")),
                start_year=_safe(e.get("Start Date")),
                end_year=_safe(e.get("End Date")),
            ))

    experience_list: list[Experience] = []
    raw_exp = _parse_json_field(row.get("Experience"))
    if isinstance(raw_exp, list):
        for e in raw_exp:
            if not isinstance(e, dict):
                continue
            experience_list.append(Experience(
                company=_safe(e.get("Company Name")),
                title=_safe(e.get("Title")),
                description=_safe(e.get("Description")),
                start_date=_safe(e.get("Start Date")),
                end_date=_safe(e.get("End Date")),
                location=_safe(e.get("Location")),
            ))

    location = _safe(row.get("Location"))
    city, country = None, None
    if location:
        parts = [p.strip() for p in location.split(",")]
        city = parts[0] if parts else None
        country = parts[-2] if len(parts) >= 3 else (parts[-1] if len(parts) >= 2 else None)

    return Person(
        name=_safe(row.get("Full Name")),
        title=_safe(row.get("Current Position Title")),
        about=_safe(row.get("About")),
        city=city,
        country_code=country,
        followers=_safe_int(row.get("LinkedIn - Followers")),
        connections=_safe_int(row.get("LinkedIn - Connections")),
        profile_url=_normalize_public_url((
            _safe(row.get("LinkedIn - Profile URL"))
            or _safe(row.get("LinkedIn - URL"))
            or _safe(row.get("Linkedin - URL"))
            or _safe(row.get("Linkedin URL"))
            or _safe(row.get("Linkedin - Profile URL"))
            or _safe(row.get("LinkedIn Profile URL"))
            or _safe(row.get("Profile URL"))
            or _safe(row.get("LinkedIn URL"))
            or _safe(row.get("URL"))
        )),
        education=education_list or None,
        experience=experience_list or None,
        educations_details=_safe(row.get("Education Level")),
    )


def _build_person_highlights(row: pd.Series) -> list[str]:
    """Extract people highlights tags."""
    raw = _parse_json_field(row.get("People Highlights"))
    if isinstance(raw, list):
        return [h.replace("_", " ").title() for h in raw if isinstance(h, str)]
    s = _safe(row.get("People Highlights"))
    if s:
        return [h.strip().replace("_", " ").title() for h in s.split(",")]
    return []


def _read_tabular(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel file into a DataFrame. No filename/type requirements."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(str(path))


def load_people(people_csv: str | Path) -> dict[str, tuple[Person, pd.Series]]:
    """Load people CSV/Excel and return a dict keyed by Specter Person ID."""
    df = _read_tabular(people_csv)
    people: dict[str, tuple[Person, pd.Series]] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("Specter - Person ID"))
        if not pid:
            continue
        people[pid] = (_parse_person(row), row)
    return people


# ---------------------------------------------------------------------------
# Company → EvidenceStore chunk builders
# ---------------------------------------------------------------------------

def _chunk(idx: int, source: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"chunk_{idx}",
        text=text.strip(),
        source_file=source,
        page_or_slide=section,
    )


def _build_overview_chunk(row: pd.Series, idx: int) -> Chunk:
    parts = [f"Company: {_safe(row.get('Company Name')) or 'Unknown'}"]
    for field, label in [
        ("Tagline", "Tagline"),
        ("Description", "Description"),
        ("Industry", "Industry"),
        ("Tech Vertical", "Tech Vertical"),
        ("Sub-industry", "Sub-industry"),
        ("Customer Focus", "Customer Focus"),
        ("Typical Customer Profile", "Typical Customer Profile"),
        ("HQ Location", "HQ Location"),
        ("Founded Date", "Founded"),
        ("Growth Stage", "Growth Stage"),
        ("Operating Status", "Operating Status"),
        ("Tags", "Tags"),
        ("Reported Clients", "Reported Clients"),
        ("Reported Traction Highlights", "Traction Highlights"),
        ("Annual Revenue Estimate (in USD)", "Annual Revenue Estimate"),
        ("Company Size", "Company Size"),
        ("Employee Count", "Employee Count"),
    ]:
        v = _safe(row.get(field))
        if v:
            parts.append(f"{label}: {v}")
    return _chunk(idx, "specter-company", "Company Overview", "\n".join(parts))


def _build_funding_chunk(row: pd.Series, idx: int) -> Chunk | None:
    parts = []
    for field, label in [
        ("Total Funding Amount (in USD)", "Total Funding"),
        ("Last Funding Amount (in USD)", "Last Round Amount"),
        ("Last Funding Date", "Last Funding Date"),
        ("Last Funding Type", "Last Funding Type"),
        ("Post Money Valuation (in USD)", "Post-Money Valuation"),
        ("Number of Funding Rounds", "Funding Rounds"),
        ("Number of Investors", "Number of Investors"),
        ("Investors", "Investors"),
    ]:
        v = _safe(row.get(field))
        if v:
            parts.append(f"{label}: {v}")

    rounds_raw = _parse_json_field(row.get("Funding Rounds"))
    if isinstance(rounds_raw, list) and rounds_raw:
        parts.append("\nFunding Round Details:")
        for r in rounds_raw:
            raised = r.get("raised")
            raised_str = f"${raised:,.0f}" if raised else "undisclosed"
            investors = ", ".join(r.get("investors", [])) or "undisclosed"
            leads = r.get("lead_investors_partners", [])
            lead_str = ""
            if leads:
                lead_names = [f"{l.get('lead_investor_name', '')} ({l.get('partner_name', '')})"
                              for l in leads if l.get("lead_investor_name")]
                if lead_names:
                    lead_str = f" | Led by: {', '.join(lead_names)}"
            parts.append(f"  - {r.get('type', 'Unknown')} ({r.get('date', 'N/A')}): "
                         f"{raised_str} from {investors}{lead_str}")

    if not parts:
        return None
    return _chunk(idx, "specter-company", "Funding & Investors",
                  f"Funding History for {_safe(row.get('Company Name')) or 'Unknown'}:\n" + "\n".join(parts))


def _build_growth_chunk(row: pd.Series, idx: int) -> Chunk | None:
    company = _safe(row.get("Company Name")) or "Unknown"
    parts = [f"Growth Metrics for {company}:"]
    has_data = False

    emp = _safe_int(row.get("Employee Count"))
    if emp:
        has_data = True
        parts.append(f"\nEmployee Count: {emp} ({_safe(row.get('Company Size')) or 'N/A'})")
        for months, field in [
            (1, "Employee Monthly Growth1"), (3, "Employee Monthly Growth3"),
            (6, "Employee Monthly Growth6"), (12, "Employee Monthly Growth12"),
            (24, "Employee Monthly Growth24"),
        ]:
            v = _safe_float(row.get(field))
            if v is not None:
                parts.append(f"  {months}mo employee growth: {v:.1f}%")

    web = _safe_int(row.get("Web Visits"))
    if web:
        has_data = True
        parts.append(f"\nMonthly Web Visits: {web:,}")
        parts.append(f"Top Country: {_safe(row.get('Top Country')) or 'N/A'}")
        traffic_src = _safe(row.get("Traffic Sources"))
        if traffic_src:
            parts.append(f"Traffic Sources: {traffic_src}")
        for months, field in [
            (1, "Web Visits Monthly Growth1"), (3, "Web Visits Monthly Growth3"),
            (6, "Web Visits Monthly Growth6"), (12, "Web Visits Monthly Growth12"),
        ]:
            v = _safe_float(row.get(field))
            if v is not None:
                parts.append(f"  {months}mo web traffic growth: {v:.1f}%")

    li_followers = _safe_int(row.get("LinkedIn Followers"))
    if li_followers:
        has_data = True
        parts.append(f"\nLinkedIn Followers: {li_followers:,}")
        for months, field in [
            (3, "Linkedin Followers Monthly Growth3"),
            (6, "Linkedin Followers Monthly Growth6"),
            (12, "Linkedin Followers Monthly Growth12"),
        ]:
            v = _safe_float(row.get(field))
            if v is not None:
                parts.append(f"  {months}mo LinkedIn growth: {v:.1f}%")

    tw = _safe_int(row.get("Twitter Followers"))
    if tw:
        has_data = True
        parts.append(f"Twitter Followers: {tw:,}")

    ig = _safe_int(row.get("Instagram Followers"))
    if ig:
        has_data = True
        parts.append(f"Instagram Followers: {ig:,}")

    downloads = _safe_int(row.get("Total App Downloads"))
    if downloads:
        has_data = True
        parts.append(f"\nTotal App Downloads: {downloads:,}")

    if not has_data:
        return None
    return _chunk(idx, "specter-company", "Growth Metrics", "\n".join(parts))


def _build_reviews_chunk(row: pd.Series, idx: int) -> Chunk | None:
    company = _safe(row.get("Company Name")) or "Unknown"
    parts = [f"Product Reviews & Ratings for {company}:"]
    has_data = False

    g2_data = _parse_json_field(row.get("G2 - Data"))
    if isinstance(g2_data, list):
        for product in g2_data:
            rating = product.get("rating")
            reviews = product.get("reviews", 0)
            if rating or reviews:
                has_data = True
                parts.append(f"\nG2: {product.get('product_name', 'Unknown')}")
                parts.append(f"  Rating: {rating}/5 ({reviews} reviews)")
                dist = product.get("star_distribution", {})
                if dist and any(v for v in dist.values() if v):
                    parts.append(f"  Stars: 5★:{dist.get('5',0)} 4★:{dist.get('4',0)} "
                                 f"3★:{dist.get('3',0)} 2★:{dist.get('2',0)} 1★:{dist.get('1',0)}")

    tp_data = _parse_json_field(row.get("Trustpilot - Data"))
    if isinstance(tp_data, dict):
        rating = tp_data.get("rating")
        reviews = tp_data.get("reviews", 0)
        if rating or reviews:
            has_data = True
            parts.append(f"\nTrustpilot: Rating {rating}/5 ({reviews} reviews)")

    for store_prefix, label in [("App Store", "App Store"), ("Google Play", "Google Play")]:
        apps_raw = _parse_json_field(row.get(f"{store_prefix} - Apps"))
        if isinstance(apps_raw, list):
            for app in apps_raw:
                rating = app.get("rating")
                reviews = app.get("reviews")
                if rating or reviews:
                    has_data = True
                    parts.append(f"\n{label}: {app.get('name', 'Unknown')}")
                    parts.append(f"  Rating: {rating}/5 ({reviews or 0} reviews)")
                    dl = app.get("downloads")
                    if dl:
                        parts.append(f"  Downloads: {dl:,}+")

    certs = _parse_json_field(row.get("Certifications"))
    if isinstance(certs, list) and certs:
        has_data = True
        parts.append(f"\nCertifications: {', '.join(str(c) for c in certs)}")

    awards = _parse_json_field(row.get("Awards"))
    if isinstance(awards, list) and awards:
        has_data = True
        parts.append("\nAwards:")
        for a in awards:
            parts.append(f"  - {a.get('award_name', 'Unknown')} ({a.get('award_year', 'N/A')}) "
                         f"by {a.get('award_org', 'N/A')}: {a.get('award_details', '')}")

    if not has_data:
        return None
    return _chunk(idx, "specter-company", "Product Reviews & Ratings", "\n".join(parts))


def _build_glassdoor_chunk(row: pd.Series, idx: int) -> Chunk | None:
    gd = _parse_json_field(row.get("Glassdoor - Data"))
    if not isinstance(gd, dict):
        return None
    reviews = gd.get("reviews_count", 0)
    rating = gd.get("ratings_overall")
    if not rating and not reviews:
        return None

    company = _safe(row.get("Company Name")) or "Unknown"
    parts = [f"Glassdoor & Culture Data for {company}:"]
    parts.append(f"Overall Rating: {rating}/5 ({reviews} reviews)")

    for key, label in [
        ("ratings_culture_values", "Culture & Values"),
        ("ratings_work_life_balance", "Work-Life Balance"),
        ("ratings_compensation_benefits", "Compensation & Benefits"),
        ("ratings_career_opportunities", "Career Opportunities"),
        ("ratings_senior_management", "Senior Management"),
    ]:
        v = gd.get(key)
        if v:
            parts.append(f"  {label}: {v}/5")

    outlook = gd.get("ratings_business_outlook")
    if outlook is not None:
        parts.append(f"  Business Outlook: {outlook*100:.0f}% positive")
    recommend = gd.get("ratings_recommend_to_friend")
    if recommend is not None:
        parts.append(f"  Recommend to Friend: {recommend*100:.0f}%")
    ceo = gd.get("ratings_ceo_approval")
    if ceo is not None:
        parts.append(f"  CEO Approval: {ceo*100:.0f}%")

    interviews = gd.get("interviews_experience")
    if isinstance(interviews, list) and interviews:
        parts.append("  Interview Experience: " +
                     ", ".join(f"{i.get('name','')}: {i.get('value','')}" for i in interviews))

    return _chunk(idx, "specter-company", "Glassdoor & Culture", "\n".join(parts))


def _build_team_overview_chunk(
    team: list[Person],
    highlights: list[str],
    company_name: str,
    num_founders: int | None,
    idx: int,
) -> Chunk:
    parts = [f"Founding Team Overview for {company_name}:"]
    parts.append(f"Number of Founders: {num_founders or len(team)}")
    if highlights:
        parts.append(f"Founder Highlights: {', '.join(highlights)}")
    parts.append("")
    for p in team:
        parts.append(p.get_profile_summary())
        parts.append("---")
    return _chunk(idx, "specter-people", "Founding Team Overview", "\n".join(parts))


def _build_person_detail_chunk(
    person: Person,
    person_row: pd.Series,
    company_name: str,
    idx: int,
) -> Chunk:
    """Build a detailed evidence chunk for a single team member."""
    name = person.name or "Unknown"
    parts = [f"Team Member Profile: {name} at {company_name}"]

    highlights = _build_person_highlights(person_row)
    if highlights:
        parts.append(f"Highlights: {', '.join(highlights)}")

    title = _safe(person_row.get("Current Position Title"))
    if title:
        parts.append(f"Current Role: {title}")

    seniority = _safe(person_row.get("Level of Seniority"))
    yoe = _safe(person_row.get("Years of Experience"))
    if seniority or yoe:
        parts.append(f"Seniority: {seniority or 'N/A'} | Years of Experience: {yoe or 'N/A'}")

    tagline = _safe(person_row.get("Tagline"))
    if tagline:
        parts.append(f"Tagline: {tagline}")

    if person.about:
        about_text = person.about[:600] if len(person.about) > 600 else person.about
        parts.append(f"About: {about_text}")

    if person.education:
        parts.append(f"Education: {person.get_education_summary()}")
    edu_raw = _parse_json_field(person_row.get("Education"))
    if isinstance(edu_raw, list):
        for e in edu_raw:
            if not isinstance(e, dict):
                continue
            field = _safe(e.get("Field of Study"))
            degree = _safe(e.get("Degree Title"))
            name_part = _safe(e.get("Name")) or "Unknown"
            if field or degree:
                extra = f"  - {name_part}: {degree or ''} {field or ''}".strip()
                parts.append(extra)

    if person.experience:
        parts.append(f"\nCareer History:")
        for exp in person.experience:
            exp_str = str(exp)
            if exp_str.strip():
                parts.append(f"  • {exp_str}")

    tenure = _safe(person_row.get("Current Tenure"))
    avg_tenure = _safe(person_row.get("Average Tenure"))
    if tenure or avg_tenure:
        parts.append(f"Current Tenure: {tenure or 'N/A'} months | Avg Tenure: {avg_tenure or 'N/A'} months")

    skills = _safe(person_row.get("Skills"))
    if skills:
        skills_parsed = _parse_json_field(skills)
        if isinstance(skills_parsed, list):
            parts.append(f"Skills: {', '.join(str(s) for s in skills_parsed[:20])}")
        else:
            parts.append(f"Skills: {skills[:300]}")

    if person.followers or person.connections:
        net_parts = []
        if person.followers:
            net_parts.append(f"{person.followers:,} LinkedIn followers")
        if person.connections:
            net_parts.append(f"{person.connections:,} connections")
        parts.append(f"Network: {', '.join(net_parts)}")

    return _chunk(idx, "specter-people", f"Team Member: {name}", "\n".join(parts))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_specter(
    companies_csv: str | Path,
    people_csv: str | Path,
) -> list[tuple[Company, EvidenceStore]]:
    """Parse Specter CSVs and return (Company, EvidenceStore) pairs.

    Args:
        companies_csv: Path to the specter-company-signals CSV.
        people_csv: Path to the specter-people-signals CSV.

    Returns:
        List of (Company, EvidenceStore) tuples, one per company row.
    """
    companies_df = _read_tabular(companies_csv)
    people_map = load_people(people_csv)

    results: list[tuple[Company, EvidenceStore]] = []

    for _, row in companies_df.iterrows():
        company_name = _safe(row.get("Company Name"))
        if not company_name:
            continue

        slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")

        # --- Join people ---
        founders_json = _parse_json_field(row.get("Founders"))
        founder_ids: list[str] = []
        if isinstance(founders_json, list):
            for f in founders_json:
                pid = f.get("specter_person_id") if isinstance(f, dict) else None
                if pid:
                    founder_ids.append(pid)

        team_persons: list[Person] = []
        team_rows: list[pd.Series] = []
        seen_ids: set[str] = set()
        for pid in founder_ids:
            if pid in people_map and pid not in seen_ids:
                person, prow = people_map[pid]
                team_persons.append(person)
                team_rows.append(prow)
                seen_ids.add(pid)

        # Also pick up people whose current company matches
        company_domain = _safe(row.get("Domain"))
        for pid, (person, prow) in people_map.items():
            if pid in seen_ids:
                continue
            current_company_website = _safe(prow.get("Current Position Company Website"))
            current_company_name = _safe(prow.get("Current Position Company Name"))
            if (
                (company_domain and current_company_website
                 and company_domain.lower() in current_company_website.lower())
                or (current_company_name
                    and current_company_name.lower() == company_name.lower())
            ):
                team_persons.append(person)
                team_rows.append(prow)
                seen_ids.add(pid)

        # --- Build Company ---
        company = Company(
            name=company_name,
            industry=_safe(row.get("Industry")),
            tagline=_safe(row.get("Tagline")),
            about=_safe(row.get("Description")),
            team=team_persons or None,
            domain=_safe(row.get("Domain")),
        )

        # --- Build EvidenceStore ---
        chunks: list[Chunk] = []
        idx = 0

        overview = _build_overview_chunk(row, idx)
        chunks.append(overview)
        idx += 1

        funding = _build_funding_chunk(row, idx)
        if funding:
            chunks.append(funding)
            idx += 1

        growth = _build_growth_chunk(row, idx)
        if growth:
            chunks.append(growth)
            idx += 1

        reviews = _build_reviews_chunk(row, idx)
        if reviews:
            chunks.append(reviews)
            idx += 1

        glassdoor = _build_glassdoor_chunk(row, idx)
        if glassdoor:
            chunks.append(glassdoor)
            idx += 1

        # Team chunks
        founder_highlights_raw = _safe(row.get("Founder Highlights"))
        founder_highlights = (
            [h.strip() for h in founder_highlights_raw.split(",")]
            if founder_highlights_raw else []
        )

        if team_persons:
            team_overview = _build_team_overview_chunk(
                team_persons, founder_highlights, company_name,
                _safe_int(row.get("Number of Founders")), idx,
            )
            chunks.append(team_overview)
            idx += 1

            for person, prow in zip(team_persons, team_rows):
                detail = _build_person_detail_chunk(person, prow, company_name, idx)
                chunks.append(detail)
                idx += 1
        elif founder_highlights:
            chunks.append(_chunk(
                idx, "specter-company", "Founder Highlights",
                f"Founder Highlights for {company_name}: {', '.join(founder_highlights)}"
            ))
            idx += 1

        store = EvidenceStore(startup_slug=slug, chunks=chunks)
        results.append((company, store))

    return results
