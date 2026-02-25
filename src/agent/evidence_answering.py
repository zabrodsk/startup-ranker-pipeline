"""Answer question trees using retrieved document chunks instead of web search.

This module replaces the web-search-based answering stage with a
retrieval-augmented approach grounded in the startup's own documents.
"""

import asyncio
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.ingest.store import EvidenceStore
from agent.llm import create_llm
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

GROUNDED_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

Evidence chunks:
{chunks_text}
"""


async def answer_question_from_evidence(
    question: str,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
) -> str:
    """Answer a single question using retrieved document chunks.

    Args:
        question: The due-diligence question to answer.
        company: Company metadata for context.
        store: EvidenceStore to retrieve from.
        k: Number of chunks to retrieve.

    Returns:
        The LLM's grounded answer string.
    """
    chunks = retrieve_chunks(question, store, k=k)

    if not chunks:
        return "Unknown from provided documents."

    chunks_text = "\n---\n".join(
        f"[{c.chunk_id}] (source: {c.source_file}, location: {c.page_or_slide}):\n{c.text}"
        for c in chunks
    )

    llm = create_llm(temperature=0.2)
    response = await llm.ainvoke([
        SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
        HumanMessage(content=GROUNDED_USER_PROMPT.format(
            company_summary=company.get_company_summary(),
            question=question,
            chunks_text=chunks_text,
        )),
    ])

    return response.content or "Unknown from provided documents."


async def _answer_node_from_evidence(
    node: QuestionNode,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
) -> None:
    """Recursively answer a question node and its children from evidence.

    Leaf nodes retrieve chunks directly. Parent nodes also retrieve chunks
    (rather than synthesizing from children) to maximize grounding.
    """
    if node.sub_nodes:
        tasks = [
            _answer_node_from_evidence(child, company, store, k)
            for child in node.sub_nodes
        ]
        await asyncio.gather(*tasks)

    node.answer = await answer_question_from_evidence(
        node.question, company, store, k
    )


async def answer_all_trees_from_evidence(
    question_trees: Dict[str, QuestionTree],
    company: Company,
    store: EvidenceStore,
    k: int = 8,
) -> List[Dict[str, str]]:
    """Answer all question trees using document evidence.

    Processes all trees in parallel, then extracts Q&A pairs in the same
    format the downstream pipeline expects.

    Args:
        question_trees: Dict mapping aspect -> QuestionTree.
        company: The Company being analyzed.
        store: EvidenceStore with ingested chunks.
        k: Chunks to retrieve per question.

    Returns:
        List of Q&A pair dicts compatible with the argument generation stage.
    """
    tasks = [
        _answer_node_from_evidence(tree.root_node, company, store, k)
        for tree in question_trees.values()
    ]
    await asyncio.gather(*tasks)

    from agent.common.utils import get_qa_pairs_from_question_tree

    all_qa_pairs: List[Dict[str, str]] = []
    for tree in question_trees.values():
        all_qa_pairs.extend(get_qa_pairs_from_question_tree(tree))

    return all_qa_pairs
