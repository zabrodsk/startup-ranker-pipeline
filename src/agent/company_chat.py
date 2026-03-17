"""Company-specific chat over persisted evidence and historical runs."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dataclasses.company import Company
from agent.evidence_answering import (
    _resolve_web_search_provider_name,
    _question_prefers_web_search,
    _web_search_domain_filter,
    WEB_RESULTS_TRUNCATE,
    WEB_SEARCH_TIMEOUT_SEC,
    _answer_indicates_no_evidence,
    _build_web_search_query,
    _coerce_text,
    _run_web_search,
    _web_results_add_value,
)
from agent.ingest.store import Chunk, EvidenceStore
from agent.llm import chat_llm_selection, create_llm
from agent.llm_catalog import serialize_selection
from agent.retrieval import retrieve_chunks
from agent.run_context import (
    PERPLEXITY_SEARCH_PRICE_PER_REQUEST_USD,
    RunTelemetryCollector,
    use_company_context,
    use_run_context,
    use_stage_context,
)

CHAT_CHUNK_PREFIX = "chunk"
CHAT_QA_PREFIX = "qa"
CHAT_ARGUMENT_PREFIX = "argument"
CHAT_TOP_K = 8
CHAT_MAX_CITATIONS = 5
CHAT_EXCERPT_CHARS = 280

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

CHAT_SYSTEM_PROMPT = """\
You are an investment analyst answering questions about a specific company.

Rules:
- Prefer the provided historical company evidence over general knowledge.
- If newer and older runs conflict, prioritize newer evidence but explicitly mention the inconsistency.
- Use web search results only when the provided evidence is insufficient.
- Keep answers concise, specific, and useful for an investment team.
- Do not fabricate facts. Say clearly when evidence is incomplete or conflicting.
"""

CHAT_GROUNDED_PROMPT = """\
Company: {company_summary}

Company run history summary:
{history_summary}

Conversation summary:
{conversation_summary}

Recent conversation:
{conversation_turns}

Current question:
{question}

Relevant evidence:
{evidence_text}

Answer the current question using the evidence above. If multiple runs disagree, state that explicitly and prefer the most recent run.
"""

CHAT_HYBRID_PROMPT = """\
Company: {company_summary}

Company run history summary:
{history_summary}

Conversation summary:
{conversation_summary}

Recent conversation:
{conversation_turns}

Current question:
{question}

Relevant evidence:
{evidence_text}

Web fallback results:
{web_results}

Answer the current question. Prefer company evidence. Use web results only to fill gaps that the evidence does not cover. If web results contradict the most recent run, say so clearly.
"""


@dataclass(slots=True)
class ChatCitation:
    kind: str
    citation_id: str
    label: str
    excerpt: str
    job_id: str | None = None
    created_at: str | None = None
    page_or_slide: str | None = None
    source_file: str | None = None
    question: str | None = None
    answer: str | None = None
    argument_type: str | None = None
    score: float | None = None
    web_search_query: str | None = None
    web_search_results: str | None = None
    web_search_provider: str | None = None
    web_search_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "citation_id": self.citation_id,
            "label": self.label,
            "excerpt": self.excerpt,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "page_or_slide": self.page_or_slide,
            "source_file": self.source_file,
            "question": self.question,
            "answer": self.answer,
            "argument_type": self.argument_type,
            "score": self.score,
            "web_search_query": self.web_search_query,
            "web_search_results": self.web_search_results,
            "web_search_provider": self.web_search_provider,
            "web_search_cost_usd": self.web_search_cost_usd,
        }


def _normalize_signature(*parts: Any) -> str:
    raw = " | ".join(str(part or "").strip().lower() for part in parts)
    return _NON_ALNUM_RE.sub(" ", raw).strip()


def _trim_excerpt(text: str, limit: int = CHAT_EXCERPT_CHARS) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _build_history_summary(runs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for run in runs[:8]:
        lines.append(
            f"- {run.get('created_at') or 'unknown date'} · job {run.get('job_id') or 'unknown'}"
            f" · {run.get('company_name') or 'Unknown company'}"
            f" · decision {run.get('decision') or 'unknown'}"
        )
    return "\n".join(lines) or "- No persisted run history available."


def _render_conversation_turns(transcript: list[dict[str, Any]]) -> str:
    if not transcript:
        return "No previous conversation."
    lines = []
    for item in transcript[-12:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {item.get('content') or ''}")
    return "\n".join(lines)


def _build_company_from_context(context: dict[str, Any]) -> Company:
    latest = (context.get("runs") or [{}])[0]
    result_payload = latest.get("results") or {}
    summary = ((result_payload.get("summary_rows") or [{}]) or [{}])[0]
    return Company(
        name=context.get("company_name") or latest.get("company_name") or "Unknown company",
        industry=result_payload.get("industry") or summary.get("industry") or "",
        tagline=result_payload.get("tagline") or "",
        about=result_payload.get("about") or "",
        domain=context.get("domain") or "",
    )


def build_company_chat_store(
    context: dict[str, Any],
) -> tuple[Company, EvidenceStore, dict[str, ChatCitation], dict[str, Any]]:
    company = _build_company_from_context(context)
    runs = sorted(
        list(context.get("runs") or []),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )
    merged_chunks: list[Chunk] = []
    citation_map: dict[str, ChatCitation] = {}
    deduped: dict[str, tuple[str, Chunk, ChatCitation, str]] = {}

    def _maybe_add(
        *,
        signature: str,
        created_at: str,
        chunk: Chunk,
        citation: ChatCitation,
    ) -> None:
        existing = deduped.get(signature)
        if existing and str(existing[3] or "") >= str(created_at or ""):
            return
        deduped[signature] = (signature, chunk, citation, created_at)

    for run in runs:
        job_id = run.get("job_id")
        created_at = str(run.get("created_at") or "")
        for chunk_row in run.get("chunks") or []:
            original_chunk_id = chunk_row.get("chunk_id") or "unknown"
            unique_id = f"{CHAT_CHUNK_PREFIX}:{job_id or 'unknown'}:{original_chunk_id}"
            text = str(chunk_row.get("text") or "")
            source_file = str(chunk_row.get("source_file") or "uploaded evidence")
            page_or_slide = str(chunk_row.get("page_or_slide") or "")
            label = f"{source_file}{f' · {page_or_slide}' if page_or_slide else ''}"
            chunk = Chunk(
                chunk_id=unique_id,
                text=text,
                source_file=f"{label} · run {job_id or 'unknown'} · {created_at or 'unknown'}",
                page_or_slide=page_or_slide or "evidence",
            )
            citation = ChatCitation(
                kind="chunk",
                citation_id=unique_id,
                label=label,
                excerpt=_trim_excerpt(text),
                job_id=job_id,
                created_at=created_at or None,
                page_or_slide=page_or_slide or None,
                source_file=source_file,
            )
            sig = _normalize_signature("chunk", source_file, page_or_slide, text)
            _maybe_add(signature=sig, created_at=created_at, chunk=chunk, citation=citation)

        result_payload = run.get("results") or {}
        for idx, qa_row in enumerate(result_payload.get("qa_provenance_rows") or []):
            question = str(qa_row.get("question") or "").strip()
            answer = str(qa_row.get("answer") or "").strip()
            if not question and not answer:
                continue
            unique_id = f"{CHAT_QA_PREFIX}:{job_id or 'unknown'}:{idx}"
            text = f"Question: {question}\nAnswer: {answer}"
            if qa_row.get("chunks_preview"):
                text += f"\nChunk preview: {qa_row.get('chunks_preview')}"
            chunk = Chunk(
                chunk_id=unique_id,
                text=text,
                source_file=f"Q&A provenance · run {job_id or 'unknown'} · {created_at or 'unknown'}",
                page_or_slide="qa",
            )
            citation = ChatCitation(
                kind="qa",
                citation_id=unique_id,
                label=question[:96] or f"Q&A from run {job_id or 'unknown'}",
                excerpt=_trim_excerpt(answer or question),
                job_id=job_id,
                created_at=created_at or None,
                question=question or None,
                answer=answer or None,
            )
            sig = _normalize_signature("qa", question, answer)
            _maybe_add(signature=sig, created_at=created_at, chunk=chunk, citation=citation)

        for idx, arg_row in enumerate(result_payload.get("argument_rows") or []):
            argument_text = str(arg_row.get("refined_text") or arg_row.get("argument_text") or "").strip()
            if not argument_text:
                continue
            argument_type = str(arg_row.get("type") or "argument").strip().lower()
            score = _coerce_score(arg_row.get("score"))
            unique_id = f"{CHAT_ARGUMENT_PREFIX}:{job_id or 'unknown'}:{idx}"
            text = f"Argument type: {argument_type}\nScore: {score if score is not None else 'n/a'}\nArgument: {argument_text}"
            if arg_row.get("critique_text"):
                text += f"\nCritique: {arg_row.get('critique_text')}"
            chunk = Chunk(
                chunk_id=unique_id,
                text=text,
                source_file=f"Argument audit · run {job_id or 'unknown'} · {created_at or 'unknown'}",
                page_or_slide="argument",
            )
            citation = ChatCitation(
                kind="argument",
                citation_id=unique_id,
                label=f"{argument_type.upper()} argument" if argument_type else "Argument",
                excerpt=_trim_excerpt(argument_text),
                job_id=job_id,
                created_at=created_at or None,
                argument_type=argument_type or None,
                score=score,
            )
            sig = _normalize_signature("argument", argument_type, argument_text)
            _maybe_add(signature=sig, created_at=created_at, chunk=chunk, citation=citation)

    for _, chunk, citation, _created_at in deduped.values():
        merged_chunks.append(chunk)
        citation_map[chunk.chunk_id] = citation

    store = EvidenceStore(startup_slug=context.get("company_lookup_key") or company.name, chunks=merged_chunks)
    metadata = {
        "run_count": len(runs),
        "source_counts": {
            "chunks": sum(1 for item in citation_map.values() if item.kind == "chunk"),
            "qa": sum(1 for item in citation_map.values() if item.kind == "qa"),
            "arguments": sum(1 for item in citation_map.values() if item.kind == "argument"),
        },
        "history_summary": _build_history_summary(runs),
    }
    return company, store, citation_map, metadata


async def answer_company_question(
    *,
    context: dict[str, Any],
    transcript: list[dict[str, Any]],
    conversation_summary: str,
    question: str,
    use_web_search: bool = True,
    active_job_id: str | None = None,
    llm_selection: dict[str, str] | None = None,
) -> dict[str, Any]:
    company, store, citation_map, meta = build_company_chat_store(context)
    selection = serialize_selection(
        (llm_selection or chat_llm_selection()).get("provider"),
        (llm_selection or chat_llm_selection()).get("model"),
    )
    collector = RunTelemetryCollector(selected_llm=selection)
    retrieved = retrieve_chunks(question, store, k=CHAT_TOP_K)
    evidence_text = "\n---\n".join(
        f"[{chunk.chunk_id}] ({chunk.source_file}):\n{chunk.text}" for chunk in retrieved
    ) or "No relevant persisted evidence found."
    local_citations = [
        citation_map[chunk.chunk_id].to_dict()
        for chunk in retrieved[:CHAT_MAX_CITATIONS]
        if chunk.chunk_id in citation_map
    ]

    conversation_turns = _render_conversation_turns(transcript)
    prompt_kwargs = {
        "company_summary": company.get_company_summary(),
        "history_summary": meta["history_summary"],
        "conversation_summary": conversation_summary or "No prior summary.",
        "conversation_turns": conversation_turns,
        "question": (
            f"{question}\n\nUser is currently inspecting run {active_job_id}."
            if active_job_id
            else question
        ),
        "evidence_text": evidence_text,
    }

    used_web_search = False
    web_search_query = None
    web_search_results = None
    citations = list(local_citations)

    with use_company_context(context.get("company_lookup_key")):
        with use_stage_context("company_chat"):
            with use_run_context(llm_selection=selection, telemetry_collector=collector):
                llm = create_llm(temperature=0.2)

                async def _invoke(prompt_text: str) -> str:
                    response = await llm.ainvoke(
                        [
                            SystemMessage(content=CHAT_SYSTEM_PROMPT),
                            HumanMessage(content=prompt_text),
                        ]
                    )
                    return _coerce_text(response.content).strip()

                answer = await _invoke(CHAT_GROUNDED_PROMPT.format(**prompt_kwargs))

                prefers_web = _question_prefers_web_search(question)
                if use_web_search and (_answer_indicates_no_evidence(answer) or prefers_web):
                    web_search_query = _build_web_search_query(company, question)
                    try:
                        raw_web_results = await asyncio.wait_for(
                            asyncio.to_thread(
                                _run_web_search,
                                web_search_query,
                                _web_search_domain_filter(company, question),
                            ),
                            timeout=WEB_SEARCH_TIMEOUT_SEC,
                        )
                        useful, reason = _web_results_add_value(question, company.name, raw_web_results)
                        if useful or (prefers_web and raw_web_results and not str(raw_web_results).lower().startswith("web search failed")):
                            used_web_search = True
                            web_search_results = raw_web_results[:WEB_RESULTS_TRUNCATE]
                            if len(raw_web_results) > WEB_RESULTS_TRUNCATE:
                                web_search_results += "\n...[truncated]"
                            answer = await _invoke(
                                CHAT_HYBRID_PROMPT.format(
                                    **prompt_kwargs,
                                    web_results=raw_web_results,
                                )
                            )
                            web_search_provider = _resolve_web_search_provider_name()
                            citations.insert(
                                0,
                                ChatCitation(
                                    kind="web",
                                    citation_id="web:fallback",
                                    label="Web fallback",
                                    excerpt=_trim_excerpt(raw_web_results),
                                    web_search_query=web_search_query,
                                    web_search_results=web_search_results,
                                    web_search_provider=web_search_provider,
                                    web_search_cost_usd=PERPLEXITY_SEARCH_PRICE_PER_REQUEST_USD
                                    if web_search_provider == "sonar"
                                    else None,
                                ).to_dict(),
                            )
                        else:
                            web_search_results = f"Skipped web fallback: {reason}"
                    except Exception as exc:
                        web_search_results = f"Web fallback failed: {exc}"

    used_run_ids = list(
        dict.fromkeys(citation.get("job_id") for citation in citations if citation.get("job_id"))
    )
    run_costs = collector.build_run_costs()
    return {
        "answer": answer or "Insufficient information available.",
        "citations": citations,
        "used_run_ids": used_run_ids,
        "used_web_search": used_web_search,
        "web_search_query": web_search_query,
        "web_search_results": web_search_results,
        "llm_selection": selection,
        "model_label": selection["label"],
        "run_costs": run_costs,
        "model_executions": collector.snapshot_model_executions(),
        "source_counts": meta["source_counts"],
        "run_count": meta["run_count"],
    }
