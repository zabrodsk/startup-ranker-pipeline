"""Answer question trees using retrieved document chunks and optional web search.

Documents (uploaded spreadsheets, pitch decks, etc.) are the PRIMARY source —
they contain insider information and drive the analysis. Web search (Perplexity)
is a FALLBACK only when the grounded answer indicates lack of evidence.
"""

import asyncio
import os
import re
from typing import Any, Callable, Dict, List

# Timeout per LLM call to avoid indefinite hangs (e.g. API stalls, rate limits)
LLM_ANSWER_TIMEOUT_SEC = 120
# Timeout for web search (Perplexity/Brave can be slow)
WEB_SEARCH_TIMEOUT_SEC = 45
# Max concurrent LLM calls to avoid rate limiting and connection exhaustion
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("LLM_MAX_CONCURRENT", "2"))
# Max web search (Perplexity/Brave) calls per company to control cost
# Default raised from 2 -> 100 to strongly favor coverage over cost.
MAX_PPLX_CALLS_PER_COMPANY = int(os.getenv("MAX_PPLX_CALLS_PER_COMPANY", "100"))
# When to trigger web search:
#   "answer" = only when LLM answer indicates no evidence (e.g. "Unknown from provided documents")
#   "no_chunks" = legacy: only when documents have zero chunks
WEB_SEARCH_TRIGGER = os.getenv("WEB_SEARCH_TRIGGER", "answer").lower()  # "answer" | "no_chunks"

# Patterns indicating the LLM could not answer from documents
_ANSWER_NO_EVIDENCE_PATTERNS = [
    r"unknown\s+from\s+provided\s+documents",
    r"no\s+information\s+(about|regarding|on)\s+",
    r"there\s+is\s+no\s+information",
    r"does\s+not\s+contain\s+(any\s+)?(information|data|evidence)",
    r"contains\s+no\s+(information|historical|data|evidence)",
    r"however,\s+there\s+is\s+no",
    r"but\s+contains\s+no\s+",
    r"insufficient\s+information\s+available",
    r"no\s+relevant\s+(document\s+)?chunks\s+found",
    r"outlines\s+.*?\s+but\s+contains\s+no\s+",  # "strategy outlines X but contains no Y"
]
_ANSWER_NO_EVIDENCE_RE = re.compile(
    "|".join(f"({p})" for p in _ANSWER_NO_EVIDENCE_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

_SEARCH_BAD_RESULT_PATTERNS = [
    r"no web search api key configured",
    r"web search failed",
    r"\b429\b",
    r"rate limit",
    r"timed out",
    r"timeout",
    r"no relevant (web )?results",
    r"no search results found",
]
_SEARCH_BAD_RESULT_RE = re.compile(
    "|".join(f"({p})" for p in _SEARCH_BAD_RESULT_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

_TOKEN_STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "does", "do", "is", "are", "the",
    "and", "or", "for", "with", "from", "that", "this", "into", "about", "their",
    "its", "they", "them", "you", "your", "can", "could", "would", "should", "built",
    "support", "supports", "product", "company",
}

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.ingest.store import EvidenceStore
from agent.llm import create_llm
from agent.prompt_library.manager import get_prompt
from agent.rate_limit import gather_with_concurrency
from agent.retrieval import retrieve_chunks

GROUNDED_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

Rules:
- Answer ONLY using the evidence chunks provided below.
- Cite chunks by their IDs, e.g. [chunk_12], [chunk_44].
- Keep answers concise (under 80 words) and data-backed.
- If the provided evidence does not contain enough information to answer, \
respond with "Unknown from provided documents."
- Do NOT invent facts or use external knowledge.
"""

HYBRID_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

You have TWO sources of information:
1. Document evidence chunks from the startup's own materials (cite as [chunk_XX]) — \
this is the PRIMARY source. Insider info from pitch decks, financials, and spreadsheets \
is critical and authoritative.
2. Web search results (cite as [web]) — ONLY a fallback when documents lack the answer.

Rules:
- Prioritize document evidence ALWAYS. It is the main source.
- Use web search results ONLY to fill gaps when documents cannot answer.
- Keep answers concise (under 100 words) and data-backed.
- Always indicate which source you are citing.
- If neither source has enough information, say "Insufficient information available."
"""

GROUNDED_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

Evidence chunks:
{chunks_text}
"""

HYBRID_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

=== Document Evidence ===
{chunks_text}

=== Web Search Results ===
{web_results}
"""


def _coerce_text(value: Any) -> str:
    """Coerce model/tool outputs to plain text.

    Some providers can return list/dict content blocks instead of a raw string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val)
                    continue
                content_val = item.get("content")
                if isinstance(content_val, str) and content_val.strip():
                    parts.append(content_val)
                    continue
            parts.append(str(item))
        return " ".join(p.strip() for p in parts if p and p.strip()).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return str(value)
    return str(value)


def _answer_indicates_no_evidence(answer: Any) -> bool:
    """Return True if the LLM answer indicates it lacks evidence from documents.

    Matches patterns: Unknown from provided documents; no information about;
    there is no information; contains no / does not contain; however, there is no;
    insufficient information available; no relevant chunks found; outlines...but contains no.
    Empty answers are treated as no evidence.
    """
    text = _coerce_text(answer)
    if not text or not text.strip():
        return True
    return bool(_ANSWER_NO_EVIDENCE_RE.search(text))


def _tokenize_text(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {t for t in tokens if len(t) >= 3 and t not in _TOKEN_STOPWORDS}


def _web_results_add_value(question: str, company_name: str, web_results: str) -> tuple[bool, str]:
    """Heuristic quality gate for fallback search results.

    Returns (is_useful, reason).
    """
    raw = (web_results or "").strip()
    if _SEARCH_BAD_RESULT_RE.search(raw):
        return (False, "web results indicate failure/noise")
    if len(raw) < 80:
        return (False, "web results too short")

    result_tokens = _tokenize_text(raw)
    if not result_tokens:
        return (False, "web results have no usable tokens")

    company_tokens = _tokenize_text(company_name)
    if company_tokens and not (company_tokens & result_tokens):
        return (False, "web results do not mention the company")

    q_tokens = _tokenize_text(question)
    if not q_tokens:
        return (True, "question has no strong tokens; accepting web results")

    overlap = len(q_tokens & result_tokens)
    overlap_ratio = overlap / max(1, len(q_tokens))
    if overlap < 2 and overlap_ratio < 0.20:
        return (False, "web results weakly related to question")

    return (True, "web results relevant to company/question")


def _run_web_search(search_query: str) -> str:
    """Run a web search using the configured provider."""
    from datetime import datetime

    provider_name = os.getenv("WEB_SEARCH_PROVIDER", "sonar")
    pplx_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")

    if provider_name == "sonar" and (not pplx_key or pplx_key == "your_perplexity_api_key_here"):
        if brave_key:
            provider_name = "brave"
        else:
            return "No web search API key configured."

    if provider_name == "brave" and not brave_key:
        if pplx_key and pplx_key != "your_perplexity_api_key_here":
            provider_name = "sonar"
        else:
            return "No web search API key configured."

    try:
        from agent.web_search import get_provider

        search_date = datetime.now().strftime("%Y-%m-%d")
        provider = get_provider(search_end_date=search_date, provider_name=provider_name)
        return provider.search(search_query)
    except Exception as exc:
        return f"Web search failed: {exc}"


CHUNK_PREVIEW_CHARS = 200
WEB_RESULTS_TRUNCATE = 8000

async def answer_question_from_evidence(
    question: str,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    web_search_state: dict | None = None,
    vc_context: str = "",
    aspect: str = "",
    prompt_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """Answer a single question using retrieved document chunks and optional web search.

    Perplexity API is used only when ALL of the following hold:
    1. use_web_search=True (user enabled the toggle)
    2. needs_search=True, where:
       - If WEB_SEARCH_TRIGGER=answer: _answer_indicates_no_evidence(grounded_answer)
         (answer matches patterns like "Unknown from provided documents", "contains no",
         "there is no information", etc.)
       - If WEB_SEARCH_TRIGGER=no_chunks: not chunks
    3. Per-company cap not exceeded (web_search_state["count"] < max)

    Flow: (1) Run grounded LLM call first. (2) If answer indicates no evidence and
    above conditions hold, call Perplexity and run hybrid LLM with web results.

    Args:
        question: The due-diligence question to answer.
        company: Company metadata for context.
        store: EvidenceStore to retrieve from.
        k: Number of chunks to retrieve.
        use_web_search: If True, may run a web search when docs have no evidence.
        semaphore: Optional semaphore to limit concurrent LLM calls.
        web_search_state: Optional dict with count, lock, max for per-company search cap.

    Returns:
        Tuple of (answer string, provenance dict with chunk_ids, chunks_preview,
        web_search_query, web_search_results).
    """
    chunks = retrieve_chunks(question, store, k=k)

    chunk_ids = [c.chunk_id for c in chunks] if chunks else []
    chunks_preview = "\n---\n".join(
        f"[{c.chunk_id}]: {c.text[:CHUNK_PREVIEW_CHARS]}{'...' if len(c.text) > CHUNK_PREVIEW_CHARS else ''}"
        for c in chunks
    ) if chunks else ""

    chunks_text = "\n---\n".join(
        f"[{c.chunk_id}] (source: {c.source_file}, location: {c.page_or_slide}):\n{c.text}"
        for c in chunks
    ) if chunks else "No relevant document chunks found."

    web_search_query: str | None = None
    web_search_results: str | None = None
    web_search_used = False
    web_search_decision = "not requested"
    grounded_system_prompt = get_prompt("evidence.grounded.system", prompt_overrides)
    grounded_user_prompt = get_prompt("evidence.grounded.user", prompt_overrides)
    hybrid_system_prompt = get_prompt("evidence.hybrid.system", prompt_overrides)
    hybrid_user_prompt = get_prompt("evidence.hybrid.user", prompt_overrides)

    vc_block = ""
    if aspect == "general_company" and vc_context:
        vc_str = vc_context if isinstance(vc_context, str) else " ".join(str(x) for x in vc_context)
        if vc_str.strip():
            vc_block = f"VC Investment Strategy (use when evaluating alignment):\n{vc_str.strip()}\n\n"

    async def _do_llm_call() -> tuple[str, dict]:
        nonlocal web_search_query, web_search_results, web_search_used, web_search_decision

        # Step 1: Always run the grounded LLM call first (documents only)
        if not chunks:
            grounded_answer = "Unknown from provided documents."
        else:
            llm = create_llm(temperature=0.2)
            user_content = grounded_user_prompt.format(
                company_summary=company.get_company_summary(),
                question=question,
                chunks_text=chunks_text,
            )
            response = await llm.ainvoke([
                SystemMessage(content=grounded_system_prompt),
                HumanMessage(content=vc_block + user_content),
            ])
            grounded_answer = _coerce_text(response.content) or "Unknown from provided documents."

        # Step 2: Decide whether to run Perplexity based on the answer
        # "answer" trigger: search only when LLM says it lacks evidence
        # "no_chunks" trigger: search only when we had no chunks (legacy)
        if WEB_SEARCH_TRIGGER == "answer":
            needs_search = use_web_search and _answer_indicates_no_evidence(grounded_answer)
        else:
            needs_search = use_web_search and not chunks
        web_search_decision = "needed" if needs_search else "not needed"

        # Per-company cap: check and increment under lock
        do_search = False
        if needs_search and web_search_state is not None:
            async with web_search_state["lock"]:
                if web_search_state["count"][0] < web_search_state["max"]:
                    web_search_state["count"][0] += 1
                    do_search = True
                else:
                    web_search_decision = "skipped: cap reached"
        elif needs_search:
            do_search = True

        if do_search:
            web_search_decision = "attempted"
            web_search_query = f"\"{company.name}\" {question}"
            web_results = await asyncio.wait_for(
                asyncio.to_thread(_run_web_search, web_search_query),
                timeout=WEB_SEARCH_TIMEOUT_SEC,
            )
            web_search_results = web_results[:WEB_RESULTS_TRUNCATE]
            if len(web_results) > WEB_RESULTS_TRUNCATE:
                web_search_results += "\n...[truncated]"

            useful, reason = _web_results_add_value(question, company.name, web_results)
            if not useful:
                web_search_decision = f"skipped: {reason}"
                provenance = {
                    "chunk_ids": chunk_ids,
                    "chunks_preview": chunks_preview,
                    "web_search_query": web_search_query,
                    "web_search_results": web_search_results,
                    "web_search_used": web_search_used,
                    "web_search_decision": web_search_decision,
                }
                return (grounded_answer, provenance)
            web_search_used = True
            web_search_decision = f"used: {reason}"

            llm = create_llm(temperature=0.2)
            user_content = hybrid_user_prompt.format(
                company_summary=company.get_company_summary(),
                question=question,
                chunks_text=chunks_text,
                web_results=web_results,
            )
            response = await llm.ainvoke([
                SystemMessage(content=hybrid_system_prompt),
                HumanMessage(content=vc_block + user_content),
            ])
            answer = _coerce_text(response.content) or "Unknown from provided documents."
            provenance = {
                "chunk_ids": chunk_ids,
                "chunks_preview": chunks_preview,
                "web_search_query": web_search_query,
                "web_search_results": web_search_results,
                "web_search_used": web_search_used,
                "web_search_decision": web_search_decision,
            }
            return (answer, provenance)

        # Return grounded answer (no web search)
        provenance = {
            "chunk_ids": chunk_ids,
            "chunks_preview": chunks_preview,
            "web_search_query": web_search_query,
            "web_search_results": web_search_results,
            "web_search_used": web_search_used,
            "web_search_decision": web_search_decision,
        }
        return (grounded_answer, provenance)

    try:
        if semaphore:
            async with semaphore:
                answer, provenance = await asyncio.wait_for(
                    _do_llm_call(),
                    timeout=LLM_ANSWER_TIMEOUT_SEC,
                )
        else:
            answer, provenance = await asyncio.wait_for(
                _do_llm_call(),
                timeout=LLM_ANSWER_TIMEOUT_SEC,
            )
        return (answer, provenance)
    except asyncio.TimeoutError:
        provenance = {
            "chunk_ids": chunk_ids,
            "chunks_preview": chunks_preview,
            "web_search_query": web_search_query,
            "web_search_results": web_search_results,
            "web_search_used": web_search_used,
            "web_search_decision": web_search_decision,
        }
        return ("Answer timed out (API slow or rate limited).", provenance)


def _count_nodes(node: QuestionNode) -> int:
    """Count total nodes in a question tree (root + all descendants)."""
    return 1 + sum(_count_nodes(c) for c in node.sub_nodes)


def _count_nodes_in_trees(trees: Dict[str, QuestionTree]) -> int:
    """Count total nodes across all question trees."""
    return sum(_count_nodes(t.root_node) for t in trees.values())


async def _answer_node_from_evidence(
    node: QuestionNode,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    progress: List[int] | None = None,
    total: int = 0,
    semaphore: asyncio.Semaphore | None = None,
    web_search_state: dict | None = None,
    vc_context: str = "",
    aspect: str = "",
    prompt_overrides: dict[str, Any] | None = None,
) -> None:
    """Recursively answer a question node and its children from evidence.

    Leaf nodes retrieve chunks directly. Parent nodes also retrieve chunks
    (rather than synthesizing from children) to maximize grounding.
    """
    if node.sub_nodes:
        tasks = [
            _answer_node_from_evidence(
                child, company, store, k, use_web_search,
                on_progress=on_progress, progress=progress, total=total,
                semaphore=semaphore,
                web_search_state=web_search_state,
                vc_context=vc_context,
                aspect=aspect,
                prompt_overrides=prompt_overrides,
            )
            for child in node.sub_nodes
        ]
        await gather_with_concurrency(tasks, limit=MAX_CONCURRENT_LLM_CALLS)

    answer, provenance = await answer_question_from_evidence(
        node.question, company, store, k, use_web_search,
        semaphore=semaphore,
        web_search_state=web_search_state,
        vc_context=vc_context,
        aspect=aspect,
        prompt_overrides=prompt_overrides,
    )
    node.answer = answer
    node.provenance = provenance

    if progress is not None and on_progress:
        progress[0] += 1
        on_progress(progress[0], total)


async def answer_all_trees_from_evidence(
    question_trees: Dict[str, QuestionTree],
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    vc_context: str = "",
    prompt_overrides: dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """Answer all question trees using document evidence and optional web search.

    Processes all trees in parallel, then extracts Q&A pairs in the same
    format the downstream pipeline expects.

    Args:
        question_trees: Dict mapping aspect -> QuestionTree.
        company: The Company being analyzed.
        store: EvidenceStore with ingested chunks.
        k: Chunks to retrieve per question.
        use_web_search: If True, supplement document evidence with web search.
        on_progress: Optional callback(current, total) called after each Q&A.

    Returns:
        List of Q&A pair dicts compatible with the argument generation stage.
    """
    total = _count_nodes_in_trees(question_trees)
    progress: List[int] = [0]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    web_search_state: dict | None = None
    if use_web_search:
        web_search_state = {
            "count": [0],
            "lock": asyncio.Lock(),
            "max": MAX_PPLX_CALLS_PER_COMPANY,
        }

    tasks = [
        _answer_node_from_evidence(
            tree.root_node, company, store, k, use_web_search,
            on_progress=on_progress, progress=progress, total=total,
            semaphore=semaphore,
            web_search_state=web_search_state,
            vc_context=vc_context,
            aspect=aspect,
            prompt_overrides=prompt_overrides,
        )
        for aspect, tree in question_trees.items()
    ]
    await gather_with_concurrency(tasks, limit=MAX_CONCURRENT_LLM_CALLS)

    from agent.common.utils import get_qa_pairs_from_question_tree

    all_qa_pairs: List[Dict[str, str]] = []
    for tree in question_trees.values():
        all_qa_pairs.extend(get_qa_pairs_from_question_tree(tree))

    return all_qa_pairs
