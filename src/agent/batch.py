"""Batch startup evaluation CLI.

Usage:
    python -m agent.batch --input ./deals --output results.xlsx
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.dataclasses.config import Config
from agent.evidence_answering import answer_all_trees_from_evidence
from agent.ingest import EvidenceStore, ingest_startup_folder
from agent.llm import create_llm
from agent.pipeline.stages.parallel_decomposition import decompose_all_questions
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState


# ---------------------------------------------------------------------------
# Company info extraction
# ---------------------------------------------------------------------------

class CompanyInfo(BaseModel):
    """LLM-extracted company metadata."""

    company_name: str = Field(description="Official company name")
    industry: str = Field(description="Primary industry or sector")
    tagline: str = Field(default="", description="One-line tagline if available")
    about: str = Field(default="", description="Brief company description (1-2 sentences)")


EXTRACT_COMPANY_PROMPT = """\
Extract the company name, industry, tagline (if any), and a brief description \
from the following document excerpts. If something is not mentioned, leave it blank.

Excerpts:
{text}
"""


async def extract_company_info(store: EvidenceStore, slug: str) -> Company:
    """Use the first chunks to extract company metadata via LLM."""
    preview_text = "\n---\n".join(
        c.text for c in store.chunks[:6]
    )[:3000]

    if not preview_text.strip():
        return Company(name=slug)

    llm = create_llm(temperature=0.0)
    llm_structured = llm.with_structured_output(CompanyInfo)

    try:
        info: CompanyInfo = await llm_structured.ainvoke([
            SystemMessage(content="You extract structured company metadata from documents."),
            HumanMessage(content=EXTRACT_COMPANY_PROMPT.format(text=preview_text)),
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
) -> Dict[str, Any]:
    """Run the full DIALECTIC pipeline for one startup folder.

    Returns a result dict with keys:
        slug, company, evidence_store, final_state
    """
    slug = folder.name
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
    print("  [1/4] Ingesting files...")
    store = ingest_startup_folder(folder)
    if not store.chunks:
        print(f"  Skipping {slug}: no extractable content found.")
        return {"slug": slug, "skipped": True}

    print(f"         {len(store.chunks)} chunks from {folder}")

    # 2. Extract company info
    print("  [2/4] Extracting company info...")
    company = await extract_company_info(store, slug)
    print(f"         Company: {company.name} | Industry: {company.industry or 'N/A'}")

    # 3. Decompose questions & answer from evidence
    print("  [3/4] Decomposing questions & answering from evidence...")
    temp_state = IterativeInvestmentStoryState(company=company, config=config)
    decomp_result = await decompose_all_questions(temp_state)
    question_trees = decomp_result["question_trees"]

    all_qa_pairs = await answer_all_trees_from_evidence(
        question_trees, company, store, k=k,
    )
    print(f"         {len(all_qa_pairs)} Q&A pairs generated")

    # 4. Run existing DIALECTIC graph (enters at argument generation)
    print("  [4/4] Running argument generation & refinement pipeline...")
    from agent.pipeline.graph import graph

    final_state = await graph.ainvoke(
        {
            "company": company,
            "config": config,
            "all_qa_pairs": all_qa_pairs,
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

    rows.sort(key=lambda r: r.get("total_score", 0), reverse=True)
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
            rows.append({
                "startup_slug": slug,
                "type": arg.argument_type,
                "score": arg.score,
                "argument_text": arg.content,
                "critique_text": arg.critique or "",
                "refined_text": arg.refined_content or "",
                "iteration": current_iteration,
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
    """Write the three-sheet Excel workbook."""
    summary = pd.DataFrame(build_summary_rows(results))
    arguments = pd.DataFrame(build_argument_rows(results))
    evidence = pd.DataFrame(build_evidence_rows(results))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        arguments.to_excel(writer, sheet_name="Arguments", index=False)
        evidence.to_excel(writer, sheet_name="Evidence", index=False)

    print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch startup evaluation using DIALECTIC pipeline",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to deals folder (each subfolder = one startup)",
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
    return parser.parse_args(argv)


async def async_main(argv: List[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

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

    results: List[Dict[str, Any]] = []
    for folder in folders:
        try:
            result = await evaluate_startup(folder, k=args.k)
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR evaluating {folder.name}: {exc}")
            results.append({"slug": folder.name, "skipped": True})

    evaluated = [r for r in results if not r.get("skipped")]
    if not evaluated:
        print("\nNo startups were successfully evaluated.")
        sys.exit(1)

    export_excel(results, args.output)
    print(f"\nDone. {len(evaluated)} startup(s) ranked in {args.output}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
