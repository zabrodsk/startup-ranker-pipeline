"""Batch startup evaluation CLI.

Usage:
    python -m agent.batch --input ./deals --output results.xlsx
    python -m agent.batch --specter-companies companies.csv --specter-people people.csv --output results.xlsx
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.dataclasses.config import Config
from agent.dataclasses.ranking import CompanyRankingResult
from agent.evidence_answering import answer_all_trees_from_evidence
from agent.ingest import EvidenceStore, ingest_startup_folder
from agent.llm import create_llm
from agent.prompt_library.manager import get_prompt
from agent.pipeline.stages.parallel_decomposition import decompose_all_questions
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState


def _ensure_str(val: Any) -> str:
    """Convert value to str; handle list (join) to avoid 'list' has no attribute 'strip'."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val) if val else ""
    return str(val)


# ---------------------------------------------------------------------------
# Company info extraction
# ---------------------------------------------------------------------------

class CompanyInfo(BaseModel):
    """LLM-extracted company metadata."""

    company_name: str = Field(description="Official company name")
    industry: str = Field(description="Primary industry or sector")
    tagline: str = Field(default="", description="One-line tagline if available")
    about: str = Field(default="", description="Brief company description (1-2 sentences)")


async def extract_company_info(
    store: EvidenceStore,
    slug: str,
    prompt_overrides: dict[str, Any] | None = None,
) -> Company:
    """Use the first chunks to extract company metadata via LLM."""
    preview_text = "\n---\n".join(
        c.text for c in store.chunks[:6]
    )[:3000]

    if not preview_text.strip():
        return Company(name=slug)

    llm = create_llm(temperature=0.0)
    llm_structured = llm.with_structured_output(CompanyInfo)

    try:
        extract_company_system_prompt = get_prompt(
            "preprocess.extract_company.system",
            prompt_overrides,
        )
        extract_company_user_prompt = get_prompt(
            "preprocess.extract_company.user",
            prompt_overrides,
        )
        info: CompanyInfo = await llm_structured.ainvoke([
            SystemMessage(content=extract_company_system_prompt),
            HumanMessage(content=extract_company_user_prompt.format(text=preview_text)),
        ])
        return Company(
            name=info.company_name or slug,
            industry=info.industry or None,
            tagline=info.tagline or None,
            about=info.about or None,
        )
    except Exception as exc:
        print(f"  Warning: company info extraction failed ({exc}), using folder name.")
        return Company(name=slug)


# ---------------------------------------------------------------------------
# Single startup pipeline
# ---------------------------------------------------------------------------

async def evaluate_startup(
    folder: Path,
    k: int = 8,
    config: Config | None = None,
    use_web_search: bool = False,
    on_progress: Callable[[str], None] | None = None,
    vc_investment_strategy: str | None = None,
    prompt_overrides: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full DIALECTIC pipeline for one startup folder.

    Args:
        folder: Path to the startup's document folder.
        k: Number of evidence chunks to retrieve per question.
        config: Pipeline configuration.
        use_web_search: If True, supplement document evidence with web search.

    Returns a result dict with keys:
        slug, company, evidence_store, final_state
    """
    slug = folder.name

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        print(f"  {msg}")

    print(f"\n{'='*60}")
    print(f"  Evaluating: {slug}")
    print(f"{'='*60}")

    if config is None:
        config = Config(
            n_pro_arguments=3,
            n_contra_arguments=3,
            k_best_arguments_per_iteration=[3, 1],
            max_iterations=1,
        )

    # 1. Ingest
    _progress("[1/4] Ingesting files & extracting content...")
    store = ingest_startup_folder(folder)
    if not store.chunks:
        print(f"  Skipping {slug}: no extractable content found.")
        return {"slug": slug, "skipped": True}

    _progress(f"[1/4] {len(store.chunks)} chunks extracted")

    # 2. Extract company info
    _progress("[2/4] Identifying company information...")
    company = await extract_company_info(
        store,
        slug,
        prompt_overrides=prompt_overrides,
    )
    _progress(f"[2/4] {company.name} | {company.industry or 'N/A'}")

    # 3. Decompose questions & answer from evidence
    _progress("[3/4] Decomposing questions & gathering evidence...")
    temp_state = IterativeInvestmentStoryState(
        company=company,
        config=config,
        prompt_overrides=prompt_overrides or {},
    )
    decomp_result = await decompose_all_questions(temp_state)
    question_trees = decomp_result["question_trees"]

    def _on_qa_progress(current: int, total: int) -> None:
        _progress(f"[3/4] Decomposing questions & gathering evidence... ({current}/{total} Q&A)")

    all_qa_pairs = await answer_all_trees_from_evidence(
        question_trees, company, store, k=k,
        use_web_search=use_web_search,
        on_progress=_on_qa_progress,
        vc_context=_ensure_str(vc_investment_strategy).strip() or "",
        prompt_overrides=prompt_overrides or {},
    )
    _progress(f"[3/4] {len(all_qa_pairs)} Q&A pairs generated")

    # 4. Run existing DIALECTIC graph (enters at argument generation)
    _progress("[4/4] Generating arguments & scoring...")
    from agent.pipeline.graph import graph

    final_state = await graph.ainvoke(
        {
            "company": company,
            "config": config,
            "all_qa_pairs": all_qa_pairs,
            "prompt_overrides": prompt_overrides or {},
            "vc_context": _ensure_str(vc_investment_strategy).strip() or "",
            "slug": slug,
        },
        config={"recursion_limit": 100},
    )

    decision = final_state.get("final_decision", "unknown")
    print(f"         Decision: {decision}")

    return {
        "slug": slug,
        "company": company,
        "evidence_store": store,
        "final_state": final_state,
        "skipped": False,
    }


async def evaluate_from_specter(
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    config: Config | None = None,
    use_web_search: bool = False,
    on_progress: Callable[[str], None] | None = None,
    vc_investment_strategy: str | None = None,
    prompt_overrides: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the DIALECTIC pipeline for a pre-parsed Specter company.

    Skips file ingestion and LLM company extraction — uses the structured
    Company and EvidenceStore built from Specter CSVs directly.
    """
    slug = store.startup_slug
    print(f"\n{'='*60}")
    print(f"  Evaluating (Specter): {company.name}")
    print(f"{'='*60}")

    if config is None:
        config = Config(
            n_pro_arguments=3,
            n_contra_arguments=3,
            k_best_arguments_per_iteration=[3, 1],
            max_iterations=1,
        )

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        print(f"  {msg}")

    _progress(f"[1/3] {len(store.chunks)} evidence chunks · {company.name} ({company.industry or 'N/A'})")
    if company.team:
        print(f"         Team: {len(company.team)} member(s)")

    _progress("[2/3] Decomposing questions & gathering evidence...")
    temp_state = IterativeInvestmentStoryState(
        company=company,
        config=config,
        prompt_overrides=prompt_overrides or {},
    )
    decomp_result = await decompose_all_questions(temp_state)
    question_trees = decomp_result["question_trees"]

    def _on_qa_progress(current: int, total: int) -> None:
        _progress(f"[2/3] Decomposing questions & gathering evidence... ({current}/{total} Q&A)")

    all_qa_pairs = await answer_all_trees_from_evidence(
        question_trees, company, store, k=k,
        use_web_search=use_web_search,
        on_progress=_on_qa_progress,
        vc_context=_ensure_str(vc_investment_strategy).strip() or "",
        prompt_overrides=prompt_overrides or {},
    )
    _progress(f"[2/3] {len(all_qa_pairs)} Q&A pairs generated")

    _progress("[3/3] Generating arguments & scoring...")
    from agent.pipeline.graph import graph

    final_state = await graph.ainvoke(
        {
            "company": company,
            "config": config,
            "all_qa_pairs": all_qa_pairs,
            "prompt_overrides": prompt_overrides or {},
            "vc_context": _ensure_str(vc_investment_strategy).strip() or "",
            "slug": slug,
        },
        config={"recursion_limit": 100},
    )

    decision = final_state.get("final_decision", "unknown")
    print(f"         Decision: {decision}")

    return {
        "slug": slug,
        "company": company,
        "evidence_store": store,
        "final_state": final_state,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Batch ranking
# ---------------------------------------------------------------------------


def rank_batch_companies(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort companies by composite score and assign rank/percentile.

    Tie-breakers (in order): min_dimension_score, avg_confidence, -critical_gaps_count.
    """
    evaluated = [r for r in results if not r.get("skipped") and r.get("final_state")]
    if not evaluated:
        return results

    def sort_key(r: Dict[str, Any]) -> tuple:
        rr: CompanyRankingResult | None = r.get("final_state", {}).get("ranking_result")
        if not rr:
            return (0.0, 0.0, 0.0, 0)
        return (
            -rr.composite_score,
            -rr.min_dimension_score,
            -rr.avg_confidence,
            rr.critical_gaps_count,
        )

    evaluated_sorted = sorted(evaluated, key=sort_key)
    n = len(evaluated_sorted)

    for i, r in enumerate(evaluated_sorted):
        rr = r.get("final_state", {}).get("ranking_result")
        if rr:
            rr.rank = i + 1
            rr.percentile = round(100.0 * (n - i) / n, 1) if n > 0 else 0.0

    skipped = [r for r in results if r.get("skipped")]
    return evaluated_sorted + skipped


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _get_top_args(
    arguments: List[Argument],
    arg_type: str,
    n: int = 3,
) -> List[Argument]:
    """Get top N arguments of a given type sorted by score descending."""
    typed = [a for a in arguments if a.argument_type == arg_type]
    typed.sort(key=lambda a: a.score, reverse=True)
    return typed[:n]


def build_summary_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Summary sheet rows from pipeline results."""
    rows: List[Dict] = []

    for r in results:
        if r.get("skipped"):
            continue

        fs = r["final_state"]
        slug = r["slug"]
        company: Company = r["company"]
        final_args: List[Argument] = fs.get("final_arguments", [])
        decision = fs.get("final_decision", "unknown")
        ranking: CompanyRankingResult | None = fs.get("ranking_result")

        pro_args = [a for a in final_args if a.argument_type == "pro"]
        contra_args = [a for a in final_args if a.argument_type == "contra"]

        avg_pro = (sum(a.score for a in pro_args) / len(pro_args)) if pro_args else 0
        avg_contra = (sum(a.score for a in contra_args) / len(contra_args)) if contra_args else 0
        total_score = avg_pro - avg_contra

        row: Dict[str, Any] = {
            "startup_slug": slug,
            "company_name": company.name,
            "decision": decision,
            "total_score": round(total_score, 2),
            "avg_pro": round(avg_pro, 2),
            "avg_contra": round(avg_contra, 2),
        }

        if ranking:
            row["rank"] = ranking.rank
            row["percentile"] = ranking.percentile
            row["composite_score"] = ranking.composite_score
            row["strategy_fit_score"] = round(ranking.strategy_fit_score, 2)
            row["team_score"] = round(ranking.team_score, 2)
            row["upside_score"] = round(ranking.upside_score, 2)
            row["bucket"] = ranking.bucket
            row["critical_gaps_count"] = ranking.critical_gaps_count
            strat_snippets = next(
                (d.evidence_snippets for d in ranking.dimension_scores if d.dimension == "strategy_fit"),
                [],
            )
            team_snippets = next(
                (d.evidence_snippets for d in ranking.dimension_scores if d.dimension == "team"),
                [],
            )
            upside_snippets = next(
                (d.evidence_snippets for d in ranking.dimension_scores if d.dimension == "upside"),
                [],
            )
            row["top_evidence_strategy"] = " | ".join(strat_snippets[:2]) if strat_snippets else ""
            row["top_evidence_team"] = " | ".join(team_snippets[:2]) if team_snippets else ""
            row["top_evidence_upside"] = " | ".join(upside_snippets[:2]) if upside_snippets else ""
            row["strategy_fit_summary"] = ranking.strategy_fit_summary or ""
            row["team_summary"] = ranking.team_summary or ""
            row["potential_summary"] = ranking.potential_summary or ""
            row["key_points"] = "\n".join(ranking.key_points) if ranking.key_points else ""
            row["red_flags"] = "\n".join(ranking.red_flags) if ranking.red_flags else ""
        else:
            row["rank"] = ""
            row["percentile"] = ""
            row["composite_score"] = ""
            row["strategy_fit_score"] = ""
            row["team_score"] = ""
            row["upside_score"] = ""
            row["bucket"] = ""
            row["critical_gaps_count"] = ""
            row["top_evidence_strategy"] = ""
            row["top_evidence_team"] = ""
            row["top_evidence_upside"] = ""
            row["strategy_fit_summary"] = ""
            row["team_summary"] = ""
            row["potential_summary"] = ""
            row["key_points"] = ""
            row["red_flags"] = ""

        top_pro = _get_top_args(final_args, "pro", 3)
        top_contra = _get_top_args(final_args, "contra", 3)

        for i in range(3):
            if i < len(top_pro):
                text = top_pro[i].refined_content or top_pro[i].content
                row[f"top_pro_{i+1}"] = text
                row[f"top_pro_{i+1}_score"] = top_pro[i].score
            else:
                row[f"top_pro_{i+1}"] = ""
                row[f"top_pro_{i+1}_score"] = ""

        for i in range(3):
            if i < len(top_contra):
                text = top_contra[i].refined_content or top_contra[i].content
                row[f"top_contra_{i+1}"] = text
                row[f"top_contra_{i+1}_score"] = top_contra[i].score
            else:
                row[f"top_contra_{i+1}"] = ""
                row[f"top_contra_{i+1}_score"] = ""

        rows.append(row)

    def _sort_key(row: Dict) -> float:
        cs = row.get("composite_score")
        if isinstance(cs, (int, float)):
            return float(cs)
        return row.get("total_score", 0)

    rows.sort(key=_sort_key, reverse=True)
    return rows


def build_argument_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Arguments sheet rows."""
    rows: List[Dict] = []

    for r in results:
        if r.get("skipped"):
            continue

        fs = r["final_state"]
        slug = r["slug"]
        final_args: List[Argument] = fs.get("final_arguments", [])
        current_iteration = fs.get("current_iteration", 0)

        for arg in final_args:
            qa_pairs_used = ""
            if arg.qa_pairs:
                qa_pairs_used = "\n---\n".join(
                    f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
                    for qa in arg.qa_pairs
                )

            rows.append({
                "startup_slug": slug,
                "type": arg.argument_type,
                "score": arg.score,
                "argument_text": arg.content,
                "critique_text": arg.critique or "",
                "refined_text": arg.refined_content or "",
                "argument_feedback": arg.argument_feedback or "",
                "qa_pairs_used": qa_pairs_used,
                "iteration": current_iteration,
            })

    return rows


def build_qa_provenance_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Q&A Provenance sheet rows with evidence and web search used per question."""
    rows: List[Dict] = []

    for r in results:
        if r.get("skipped"):
            continue

        fs = r["final_state"]
        slug = r["slug"]
        company: Company = r["company"]
        all_qa_pairs = fs.get("all_qa_pairs", [])

        for qa in all_qa_pairs:
            chunk_ids = qa.get("chunk_ids")
            if isinstance(chunk_ids, list):
                chunk_ids_str = ", ".join(str(c) for c in chunk_ids)
            else:
                chunk_ids_str = str(chunk_ids) if chunk_ids else ""

            rows.append({
                "startup_slug": slug,
                "company_name": company.name,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "chunk_ids": chunk_ids_str,
                "chunks_preview": qa.get("chunks_preview", ""),
                "web_search_query": qa.get("web_search_query") or "",
                "web_search_results": qa.get("web_search_results") or "",
                "web_search_used": bool(qa.get("web_search_used")),
                "web_search_decision": qa.get("web_search_decision") or "",
            })

    return rows


def build_failed_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Failed sheet rows for companies that raised exceptions."""
    rows: List[Dict] = []
    for r in results:
        if not r.get("skipped"):
            continue
        rows.append({
            "startup_slug": r.get("slug", ""),
            "company_name": r.get("company_name", ""),
            "error": r.get("error", "Unknown error"),
        })
    return rows


def build_ranking_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Ranking sheet rows with full dimension breakdown and evidence snippets."""
    rows: List[Dict] = []

    for r in results:
        if r.get("skipped"):
            continue

        fs = r["final_state"]
        slug = r["slug"]
        company: Company = r["company"]
        ranking: CompanyRankingResult | None = fs.get("ranking_result")

        if not ranking:
            continue

        for dim_score in ranking.dimension_scores:
            rows.append({
                "startup_slug": slug,
                "company_name": company.name,
                "rank": ranking.rank,
                "composite_score": ranking.composite_score,
                "bucket": ranking.bucket,
                "dimension": dim_score.dimension,
                "raw_score": dim_score.raw_score,
                "adjusted_score": dim_score.adjusted_score,
                "confidence": dim_score.confidence,
                "evidence_count": dim_score.evidence_count,
                "evidence_snippets": " | ".join(dim_score.evidence_snippets),
                "critical_gaps": "; ".join(dim_score.critical_gaps),
            })

    return rows


def build_evidence_rows(results: List[Dict[str, Any]]) -> List[Dict]:
    """Build Evidence sheet rows."""
    rows: List[Dict] = []

    for r in results:
        if r.get("skipped"):
            continue

        store: EvidenceStore = r["evidence_store"]
        slug = r["slug"]

        for chunk in store.chunks:
            rows.append({
                "startup_slug": slug,
                "source_file": chunk.source_file,
                "page_or_slide": chunk.page_or_slide,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.text[:500],
            })

    return rows


def export_excel(results: List[Dict[str, Any]], output_path: str) -> None:
    """Write the Excel workbook (Summary, Arguments, Evidence, Q&A Provenance, Ranking, and Failed if any)."""
    results = rank_batch_companies(results)
    summary = pd.DataFrame(build_summary_rows(results))
    arguments = pd.DataFrame(build_argument_rows(results))
    evidence = pd.DataFrame(build_evidence_rows(results))
    qa_provenance = pd.DataFrame(build_qa_provenance_rows(results))
    failed = pd.DataFrame(build_failed_rows(results))

    ranking = pd.DataFrame(build_ranking_rows(results))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        arguments.to_excel(writer, sheet_name="Arguments", index=False)
        evidence.to_excel(writer, sheet_name="Evidence", index=False)
        if not qa_provenance.empty:
            qa_provenance.to_excel(writer, sheet_name="Q&A Provenance", index=False)
        if not ranking.empty:
            ranking.to_excel(writer, sheet_name="Ranking", index=False)
        if not failed.empty:
            failed.to_excel(writer, sheet_name="Failed", index=False)

    print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch startup evaluation using DIALECTIC pipeline",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to deals folder (each subfolder = one startup)",
    )
    input_group.add_argument(
        "--specter-companies",
        help="Path to Specter company-signals CSV",
    )

    parser.add_argument(
        "--specter-people",
        help="Path to Specter people-signals CSV (required with --specter-companies)",
    )
    parser.add_argument(
        "--output", default="results.xlsx",
        help="Output Excel file path (default: results.xlsx)",
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help="Number of evidence chunks to retrieve per question (default: 8)",
    )
    parser.add_argument(
        "--max-startups", type=int, default=None,
        help="Limit the number of startups to evaluate",
    )
    parser.add_argument(
        "--web-search", action="store_true", default=False,
        help="Supplement document evidence with web search results",
    )
    parser.add_argument(
        "--vc-strategy", type=str, default="",
        help="VC investment strategy text for strategy fit scoring",
    )
    return parser.parse_args(argv)


async def _run_specter_batch(args: argparse.Namespace) -> None:
    """Run batch evaluation from Specter CSV inputs."""
    from agent.ingest.specter_ingest import ingest_specter

    if not args.specter_people:
        print("Error: --specter-people is required with --specter-companies")
        sys.exit(1)

    companies_path = Path(args.specter_companies)
    people_path = Path(args.specter_people)

    if not companies_path.exists():
        print(f"Error: companies file '{companies_path}' not found.")
        sys.exit(1)
    if not people_path.exists():
        print(f"Error: people file '{people_path}' not found.")
        sys.exit(1)

    print("Parsing Specter CSV files...")
    company_store_pairs = ingest_specter(companies_path, people_path)

    if not company_store_pairs:
        print("No companies found in Specter data.")
        sys.exit(1)

    if args.max_startups:
        company_store_pairs = company_store_pairs[: args.max_startups]

    print(f"Found {len(company_store_pairs)} company(ies) to evaluate.\n")

    vc_strategy = getattr(args, "vc_strategy", "") or ""

    results: List[Dict[str, Any]] = []
    for company, store in company_store_pairs:
        try:
            result = await evaluate_from_specter(
                company, store, k=args.k, use_web_search=args.web_search,
                vc_investment_strategy=vc_strategy,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR evaluating {company.name}: {exc}")
            results.append({
                "slug": store.startup_slug,
                "skipped": True,
                "error": str(exc)[:500],
                "company_name": company.name,
            })

    evaluated = [r for r in results if not r.get("skipped")]
    if not evaluated:
        print("\nNo startups were successfully evaluated.")
        sys.exit(1)

    export_excel(results, args.output)
    print(f"\nDone. {len(evaluated)} startup(s) ranked in {args.output}")


async def _run_folder_batch(args: argparse.Namespace) -> None:
    """Run batch evaluation from document folders."""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)

    folders = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not folders:
        print(f"No startup folders found in '{input_dir}'.")
        sys.exit(1)

    if args.max_startups:
        folders = folders[: args.max_startups]

    print(f"Found {len(folders)} startup(s) to evaluate.")

    vc_strategy = getattr(args, "vc_strategy", "") or ""

    results: List[Dict[str, Any]] = []
    for folder in folders:
        try:
            result = await evaluate_startup(
                folder, k=args.k, use_web_search=args.web_search,
                vc_investment_strategy=vc_strategy,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR evaluating {folder.name}: {exc}")
            results.append({
                "slug": folder.name,
                "skipped": True,
                "error": str(exc)[:500],
                "company_name": folder.name,
            })

    evaluated = [r for r in results if not r.get("skipped")]
    if not evaluated:
        print("\nNo startups were successfully evaluated.")
        sys.exit(1)

    export_excel(results, args.output)
    print(f"\nDone. {len(evaluated)} startup(s) ranked in {args.output}")


async def async_main(argv: List[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    if args.specter_companies:
        await _run_specter_batch(args)
    else:
        await _run_folder_batch(args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
