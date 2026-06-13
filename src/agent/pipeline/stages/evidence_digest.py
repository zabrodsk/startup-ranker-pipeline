"""Optional evidence-corpus digest for critique/refinement (Sprint 3 W3).

When PIPELINE_EVIDENCE_DIGEST=on, one LLM call summarizes the full Q&A corpus
into a compact digest. Critique and refinement then receive the digest plus
only the evidence directly cited by each argument instead of the full corpus,
cutting per-argument prompt tokens. Fail-open: any error returns None and the
stages fall back to the legacy full-corpus rendering.
"""

import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.common.llm_config import get_llm
from agent.common.utils import format_qa_pairs_without_index
from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.prompt_library.manager import get_prompt
from agent.pipeline.utils.phase_llm import ainvoke_with_phase_fallback
from agent.run_context import get_current_pipeline_policy, use_stage_context

logger = logging.getLogger(__name__)

# Evidence-digest mode (Sprint 3 W3):
#   "off" = legacy behavior: critique/refinement see the full Q&A corpus (default)
#   "on"  = one digest LLM call per job; critique/refinement see the digest
#           plus only the evidence cited by each argument
# Read at call time (not import time) so tests and rollout can flip it via env.
PIPELINE_EVIDENCE_DIGEST_ENV = "PIPELINE_EVIDENCE_DIGEST"
_EVIDENCE_DIGEST_MODES = ("off", "on")
_DEFAULT_EVIDENCE_DIGEST_MODE = "off"
_WARNED_INVALID_DIGEST_MODE: set[str] = set()

_DIGEST_HEADER = "=== Evidence corpus digest ==="
_CITED_HEADER = "=== Evidence directly cited by this argument ==="


def _evidence_digest_mode() -> str:
    """Return the active evidence-digest mode."""
    raw = os.getenv(PIPELINE_EVIDENCE_DIGEST_ENV, _DEFAULT_EVIDENCE_DIGEST_MODE)
    mode = raw.strip().lower()
    if mode not in _EVIDENCE_DIGEST_MODES:
        if mode not in _WARNED_INVALID_DIGEST_MODE:
            _WARNED_INVALID_DIGEST_MODE.add(mode)
            logger.warning(
                "Invalid %s=%r; falling back to %r",
                PIPELINE_EVIDENCE_DIGEST_ENV, raw, _DEFAULT_EVIDENCE_DIGEST_MODE,
            )
        return _DEFAULT_EVIDENCE_DIGEST_MODE
    return mode


def is_evidence_digest_enabled() -> bool:
    """Return True when the evidence-digest flag is on."""
    return _evidence_digest_mode() == "on"


async def maybe_build_evidence_digest(
    all_qa_pairs: list[dict[str, Any]],
    company: Company,
    prompt_overrides: dict | None = None,
) -> str | None:
    """Build the corpus digest, or None when off/empty/failed (fail-open)."""
    if not is_evidence_digest_enabled():
        return None
    if not all_qa_pairs:
        return None

    try:
        system_prompt = get_prompt("evidence_digest.system", prompt_overrides)
        user_prompt = get_prompt("evidence_digest.user", prompt_overrides)
        policy = get_current_pipeline_policy()

        async def _invoke() -> str:
            with use_stage_context("evidence_digest"):
                llm = get_llm(temperature=0.0)
                response = await llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=user_prompt.format(
                                company_name=company.name,
                                qa_corpus=format_qa_pairs_without_index(all_qa_pairs),
                            )
                        ),
                    ]
                )
                return str(getattr(response, "content", "") or "")

        digest = await ainvoke_with_phase_fallback(
            policy.answering if policy else None,
            _invoke,
        )
        return digest.strip() or None
    except Exception:
        logger.warning(
            "Evidence digest generation failed; stages fall back to the full corpus",
            exc_info=True,
        )
        return None


def compose_critique_evidence(digest: str, argument: Argument) -> str:
    """Digest + the argument's own cited Q&A pairs (critique rendering)."""
    cited = format_qa_pairs_without_index(argument.qa_pairs)
    return f"{_DIGEST_HEADER}\n{digest}\n\n{_CITED_HEADER}\n{cited}"


def compose_refinement_evidence(
    digest: str,
    argument: Argument,
    all_qa_pairs: list[dict[str, Any]],
) -> str:
    """Digest + the argument's cited Q&A pairs rendered with GLOBAL indices.

    Refinement output (refined_qa_indices) must index into state.all_qa_pairs,
    so the cited evidence keeps the corpus-global index of each pair.
    """
    cited = "\n".join(
        f"{index}: {all_qa_pairs[index]['question']}\n{all_qa_pairs[index]['answer']}"
        for index in argument.qa_indices
        if 0 <= index < len(all_qa_pairs)
    )
    return f"{_DIGEST_HEADER}\n{digest}\n\n{_CITED_HEADER}\n{cited}"
