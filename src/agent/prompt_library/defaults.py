"""Default prompt catalog for runtime-editable evaluation prompts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.pipeline.stages.constants import INVESTMENT_QUESTIONS
from agent.prompts.argument_critique import (
    DEVILS_ADVOCATE_CONTRA_SYSTEM_PROMPT,
    DEVILS_ADVOCATE_INDIVIDUAL_CONTRA_ARGUMENT_USER_PROMPT,
    DEVILS_ADVOCATE_INDIVIDUAL_PRO_ARGUMENT_USER_PROMPT,
    DEVILS_ADVOCATE_PRO_SYSTEM_PROMPT,
)
from agent.prompts.argument_evaluation import (
    CRITERIA_MAPPING,
    EVALUATE_SINGLE_ARGUMENT_USER_PROMPT,
    SINGLE_ARGUMENT_EVALUATION_SYSTEM_PROMPT,
)
from agent.prompts.argument_generation import (
    ARGUMENT_GENERATION_SYSTEM_PROMPT,
    CONTRA_ARGUMENTS_USER_PROMPT,
    PRO_ARGUMENTS_USER_PROMPT,
)
from agent.prompts.argument_refinement import (
    REFINE_CONTRA_ARGUMENT_SYSTEM_PROMPT,
    REFINE_CONTRA_ARGUMENTS_USER_PROMPT,
    REFINE_PRO_ARGUMENT_SYSTEM_PROMPT,
    REFINE_PRO_ARGUMENTS_USER_PROMPT,
)
from agent.prompts.question_answering import (
    DECOMPOSE_QUESTION_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
)
from agent.prompts.ranking import (
    EXECUTIVE_SUMMARY_SYSTEM,
    EXECUTIVE_SUMMARY_USER,
    RANKING_STRATEGY_FIT_SYSTEM,
    RANKING_STRATEGY_FIT_USER,
    RANKING_TEAM_SYSTEM,
    RANKING_TEAM_USER,
    RANKING_UPSIDE_SYSTEM,
    RANKING_UPSIDE_USER,
)

SCHEMA_VERSION = 1

EVIDENCE_GROUNDED_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

Rules:
- Answer ONLY using the evidence chunks provided below.
- Cite chunks by their IDs, e.g. [chunk_12], [chunk_44].
- Keep answers concise (under 80 words) and data-backed.
- If the provided evidence does not contain enough information to answer, \
respond with "Unknown from provided documents."
- Do NOT invent facts or use external knowledge.
"""

EVIDENCE_HYBRID_SYSTEM_PROMPT = """\
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

# Variant selected at the call site when RDI_WEB_EVIDENCE_PLANNER=on
# (evidence_answering.py). Web evidence is a legitimate primary source for
# external questions, but sector evidence must never be converted into
# company-specific facts.
EVIDENCE_HYBRID_WEB_PLANNER_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

You have TWO sources of information:
1. Document evidence chunks from the startup's own materials (cite as [chunk_XX]) — \
the PRIMARY and authoritative source for company-internal facts (financials, \
traction, team, product details).
2. Web search results (cite as [web]) — a LEGITIMATE primary source for external \
context: market size and growth, sector dynamics, regulation, competitors, \
customer demand, geography, and technology validation.

Rules:
- For company-internal facts, rely on document evidence; if documents lack the \
fact, say so directly — do NOT substitute web speculation.
- For market/regulatory/competitive/customer/technology questions, treat web \
evidence as valid evidence even when it does not mention the company.
- When web evidence is sector-level, clearly label it as sector-level evidence. \
Never convert sector evidence into company-specific claims (e.g. do not infer \
the company's traction from sector growth).
- Keep answers concise (under 100 words) and data-backed.
- Always indicate which source you are citing.
- If neither source has enough information, say "Insufficient information available."

Good pattern: "Company-specific evidence is limited. Sector-level evidence shows \
the market is growing [web]; this supports category attractiveness but does not \
prove the company's traction."
"""

EVIDENCE_GROUNDED_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

Evidence chunks:
{chunks_text}
"""

EVIDENCE_HYBRID_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

=== Document Evidence ===
{chunks_text}

=== Web Search Results ===
{web_results}
"""

EXTRACT_COMPANY_SYSTEM_PROMPT = "You extract structured company metadata from documents."

EXTRACT_COMPANY_USER_PROMPT = """\
Extract the company name, industry, tagline (if any), and a brief description \
from the following document excerpts. If something is not mentioned, leave it blank.

Excerpts:
{text}
"""

ORDERED_PROMPT_IDS = [
    "questions.general_company",
    "questions.market",
    "questions.product",
    "questions.team",
    "decomposition.system",
    "decomposition.user",
    "evidence.grounded.system",
    "evidence.grounded.user",
    "evidence.hybrid.system",
    "evidence.hybrid.system.web_planner",
    "evidence.hybrid.user",
    "generation.system",
    "generation.pro_user",
    "generation.contra_user",
    "critique.pro_system",
    "critique.pro_user",
    "critique.contra_system",
    "critique.contra_user",
    "evaluation.criteria_mapping",
    "evaluation.system",
    "evaluation.user",
    "refinement.pro_system",
    "refinement.pro_user",
    "refinement.contra_system",
    "refinement.contra_user",
    "preprocess.extract_company.system",
    "preprocess.extract_company.user",
    "ranking.strategy_fit.system",
    "ranking.strategy_fit.user",
    "ranking.team.system",
    "ranking.team.user",
    "ranking.upside.system",
    "ranking.upside.user",
    "ranking.executive_summary.system",
    "ranking.executive_summary.user",
]

PROMPT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "questions.general_company": {
        "title": "General Company Alignment Question",
        "stage": "questions",
        "category": "Root Questions",
        "source_path": "src/agent/pipeline/stages/constants.py",
        "description": "Top-level question for strategy, stage, and geography fit.",
        "type": "text",
        "required_placeholders": [],
        "default_value": INVESTMENT_QUESTIONS["general_company"],
    },
    "questions.market": {
        "title": "Market Question",
        "stage": "questions",
        "category": "Root Questions",
        "source_path": "src/agent/pipeline/stages/constants.py",
        "description": "Top-level question for market size, growth, and unmet needs.",
        "type": "text",
        "required_placeholders": [],
        "default_value": INVESTMENT_QUESTIONS["market"],
    },
    "questions.product": {
        "title": "Product Question",
        "stage": "questions",
        "category": "Root Questions",
        "source_path": "src/agent/pipeline/stages/constants.py",
        "description": "Top-level question for product features and defensibility.",
        "type": "text",
        "required_placeholders": [],
        "default_value": INVESTMENT_QUESTIONS["product"],
    },
    "questions.team": {
        "title": "Team Question",
        "stage": "questions",
        "category": "Root Questions",
        "source_path": "src/agent/pipeline/stages/constants.py",
        "description": "Top-level question for founder and team quality.",
        "type": "text",
        "required_placeholders": [],
        "default_value": INVESTMENT_QUESTIONS["team"],
    },
    "decomposition.system": {
        "title": "Decomposition System Prompt",
        "stage": "decomposition",
        "category": "Question Decomposition",
        "source_path": "src/agent/prompts/question_answering.py",
        "description": "System instructions for generating hierarchical sub-question trees.",
        "type": "text",
        "required_placeholders": [],
        "default_value": DECOMPOSE_SYSTEM_PROMPT,
    },
    "decomposition.user": {
        "title": "Decomposition User Prompt",
        "stage": "decomposition",
        "category": "Question Decomposition",
        "source_path": "src/agent/prompts/question_answering.py",
        "description": "Template used to decompose each root question by industry.",
        "type": "text",
        "required_placeholders": ["{question}", "{industry}"],
        "default_value": DECOMPOSE_QUESTION_PROMPT,
    },
    "evidence.grounded.system": {
        "title": "Evidence Grounded System Prompt",
        "stage": "evidence",
        "category": "Evidence Answering",
        "source_path": "src/agent/evidence_answering.py",
        "description": "System prompt for document-only grounded answers.",
        "type": "text",
        "required_placeholders": [],
        "default_value": EVIDENCE_GROUNDED_SYSTEM_PROMPT,
    },
    "evidence.grounded.user": {
        "title": "Evidence Grounded User Prompt",
        "stage": "evidence",
        "category": "Evidence Answering",
        "source_path": "src/agent/evidence_answering.py",
        "description": "User prompt template for grounded answers using retrieved chunks.",
        "type": "text",
        "required_placeholders": ["{company_summary}", "{question}", "{chunks_text}"],
        "default_value": EVIDENCE_GROUNDED_USER_PROMPT,
    },
    "evidence.hybrid.system": {
        "title": "Evidence Hybrid System Prompt",
        "stage": "evidence",
        "category": "Evidence Answering",
        "source_path": "src/agent/evidence_answering.py",
        "description": "System prompt for document + web fallback answering.",
        "type": "text",
        "required_placeholders": [],
        "default_value": EVIDENCE_HYBRID_SYSTEM_PROMPT,
    },
    "evidence.hybrid.system.web_planner": {
        "title": "Evidence Hybrid System Prompt (WebEvidencePlanner)",
        "stage": "evidence",
        "category": "Evidence Answering",
        "source_path": "src/agent/evidence_answering.py",
        "description": (
            "Hybrid system prompt used when RDI_WEB_EVIDENCE_PLANNER=on: web "
            "evidence is a legitimate primary source for external (market/"
            "regulation/competitor/customer/technology) questions."
        ),
        "type": "text",
        "required_placeholders": [],
        "default_value": EVIDENCE_HYBRID_WEB_PLANNER_SYSTEM_PROMPT,
    },
    "evidence.hybrid.user": {
        "title": "Evidence Hybrid User Prompt",
        "stage": "evidence",
        "category": "Evidence Answering",
        "source_path": "src/agent/evidence_answering.py",
        "description": "User prompt template for hybrid document and web contexts.",
        "type": "text",
        "required_placeholders": [
            "{company_summary}",
            "{question}",
            "{chunks_text}",
            "{web_results}",
        ],
        "default_value": EVIDENCE_HYBRID_USER_PROMPT,
    },
    "generation.system": {
        "title": "Argument Generation System Prompt",
        "stage": "generation",
        "category": "Argument Generation",
        "source_path": "src/agent/prompts/argument_generation.py",
        "description": "System instruction for generating investment arguments.",
        "type": "text",
        "required_placeholders": [],
        "default_value": ARGUMENT_GENERATION_SYSTEM_PROMPT,
    },
    "generation.pro_user": {
        "title": "Pro Argument User Prompt",
        "stage": "generation",
        "category": "Argument Generation",
        "source_path": "src/agent/prompts/argument_generation.py",
        "description": "Template for generating pro-investment arguments.",
        "type": "text",
        "required_placeholders": ["{n_pro_arguments}", "{questions_and_answers}"],
        "default_value": PRO_ARGUMENTS_USER_PROMPT,
    },
    "generation.contra_user": {
        "title": "Contra Argument User Prompt",
        "stage": "generation",
        "category": "Argument Generation",
        "source_path": "src/agent/prompts/argument_generation.py",
        "description": "Template for generating contra-investment arguments.",
        "type": "text",
        "required_placeholders": ["{n_contra_arguments}", "{questions_and_answers}"],
        "default_value": CONTRA_ARGUMENTS_USER_PROMPT,
    },
    "critique.pro_system": {
        "title": "Critique Pro System Prompt",
        "stage": "critique",
        "category": "Devil's Advocate Critique",
        "source_path": "src/agent/prompts/argument_critique.py",
        "description": "System role for critiquing pro arguments.",
        "type": "text",
        "required_placeholders": [],
        "default_value": DEVILS_ADVOCATE_PRO_SYSTEM_PROMPT,
    },
    "critique.pro_user": {
        "title": "Critique Pro User Prompt",
        "stage": "critique",
        "category": "Devil's Advocate Critique",
        "source_path": "src/agent/prompts/argument_critique.py",
        "description": "Template for critiquing pro arguments using Q&A.",
        "type": "text",
        "required_placeholders": ["{questions_and_answers}", "{argument}"],
        "default_value": DEVILS_ADVOCATE_INDIVIDUAL_PRO_ARGUMENT_USER_PROMPT,
    },
    "critique.contra_system": {
        "title": "Critique Contra System Prompt",
        "stage": "critique",
        "category": "Devil's Advocate Critique",
        "source_path": "src/agent/prompts/argument_critique.py",
        "description": "System role for critiquing contra arguments.",
        "type": "text",
        "required_placeholders": [],
        "default_value": DEVILS_ADVOCATE_CONTRA_SYSTEM_PROMPT,
    },
    "critique.contra_user": {
        "title": "Critique Contra User Prompt",
        "stage": "critique",
        "category": "Devil's Advocate Critique",
        "source_path": "src/agent/prompts/argument_critique.py",
        "description": "Template for critiquing contra arguments using Q&A.",
        "type": "text",
        "required_placeholders": ["{questions_and_answers}", "{argument}"],
        "default_value": DEVILS_ADVOCATE_INDIVIDUAL_CONTRA_ARGUMENT_USER_PROMPT,
    },
    "evaluation.criteria_mapping": {
        "title": "Evaluation Criteria Mapping",
        "stage": "evaluation",
        "category": "Argument Evaluation",
        "source_path": "src/agent/prompts/argument_evaluation.py",
        "description": "List of 14 criterion names used in evaluation and feedback formatting.",
        "type": "list",
        "required_placeholders": [],
        "default_value": list(CRITERIA_MAPPING),
    },
    "evaluation.system": {
        "title": "Evaluation System Prompt",
        "stage": "evaluation",
        "category": "Argument Evaluation",
        "source_path": "src/agent/prompts/argument_evaluation.py",
        "description": "System instruction for scoring arguments on 14 criteria.",
        "type": "text",
        "required_placeholders": [],
        "default_value": SINGLE_ARGUMENT_EVALUATION_SYSTEM_PROMPT,
    },
    "evaluation.user": {
        "title": "Evaluation User Prompt",
        "stage": "evaluation",
        "category": "Argument Evaluation",
        "source_path": "src/agent/prompts/argument_evaluation.py",
        "description": "Template for evaluating a single argument and optional critique context.",
        "type": "text",
        "required_placeholders": ["{argument}", "{critique}"],
        "default_value": EVALUATE_SINGLE_ARGUMENT_USER_PROMPT,
    },
    "refinement.pro_system": {
        "title": "Refinement Pro System Prompt",
        "stage": "refinement",
        "category": "Argument Refinement",
        "source_path": "src/agent/prompts/argument_refinement.py",
        "description": "System instruction for refining pro arguments.",
        "type": "text",
        "required_placeholders": [],
        "default_value": REFINE_PRO_ARGUMENT_SYSTEM_PROMPT,
    },
    "refinement.pro_user": {
        "title": "Refinement Pro User Prompt",
        "stage": "refinement",
        "category": "Argument Refinement",
        "source_path": "src/agent/prompts/argument_refinement.py",
        "description": "Template for refining pro arguments with score feedback.",
        "type": "text",
        "required_placeholders": ["{questions_and_answers}", "{argument}", "{argument_feedback}"],
        "default_value": REFINE_PRO_ARGUMENTS_USER_PROMPT,
    },
    "refinement.contra_system": {
        "title": "Refinement Contra System Prompt",
        "stage": "refinement",
        "category": "Argument Refinement",
        "source_path": "src/agent/prompts/argument_refinement.py",
        "description": "System instruction for refining contra arguments.",
        "type": "text",
        "required_placeholders": [],
        "default_value": REFINE_CONTRA_ARGUMENT_SYSTEM_PROMPT,
    },
    "refinement.contra_user": {
        "title": "Refinement Contra User Prompt",
        "stage": "refinement",
        "category": "Argument Refinement",
        "source_path": "src/agent/prompts/argument_refinement.py",
        "description": "Template for refining contra arguments with score feedback.",
        "type": "text",
        "required_placeholders": ["{questions_and_answers}", "{argument}", "{argument_feedback}"],
        "default_value": REFINE_CONTRA_ARGUMENTS_USER_PROMPT,
    },
    "preprocess.extract_company.system": {
        "title": "Company Extraction System Prompt",
        "stage": "preprocess",
        "category": "Pre-Evaluation Metadata Extraction",
        "source_path": "src/agent/batch.py",
        "description": "System instruction for extracting company metadata from documents.",
        "type": "text",
        "required_placeholders": [],
        "default_value": EXTRACT_COMPANY_SYSTEM_PROMPT,
    },
    "preprocess.extract_company.user": {
        "title": "Company Extraction User Prompt",
        "stage": "preprocess",
        "category": "Pre-Evaluation Metadata Extraction",
        "source_path": "src/agent/batch.py",
        "description": "Template for extracting company metadata fields from text excerpts.",
        "type": "text",
        "required_placeholders": ["{text}"],
        "default_value": EXTRACT_COMPANY_USER_PROMPT,
    },
    "ranking.strategy_fit.system": {
        "title": "Ranking Strategy Fit System Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "System prompt for scoring VC strategy alignment.",
        "type": "text",
        "required_placeholders": [],
        "default_value": RANKING_STRATEGY_FIT_SYSTEM,
    },
    "ranking.strategy_fit.user": {
        "title": "Ranking Strategy Fit User Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "User prompt template for strategy fit scoring.",
        "type": "text",
        "required_placeholders": ["{company_summary}", "{vc_context}", "{qa_block}"],
        "default_value": RANKING_STRATEGY_FIT_USER,
    },
    "ranking.team.system": {
        "title": "Ranking Team System Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "System prompt for scoring team quality.",
        "type": "text",
        "required_placeholders": [],
        "default_value": RANKING_TEAM_SYSTEM,
    },
    "ranking.team.user": {
        "title": "Ranking Team User Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "User prompt template for team scoring.",
        "type": "text",
        "required_placeholders": ["{company_summary}", "{qa_block}"],
        "default_value": RANKING_TEAM_USER,
    },
    "ranking.upside.system": {
        "title": "Ranking Upside System Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "System prompt for scoring problem/upside potential.",
        "type": "text",
        "required_placeholders": [],
        "default_value": RANKING_UPSIDE_SYSTEM,
    },
    "ranking.upside.user": {
        "title": "Ranking Upside User Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "User prompt template for upside scoring.",
        "type": "text",
        "required_placeholders": ["{company_summary}", "{qa_block}"],
        "default_value": RANKING_UPSIDE_USER,
    },
    "ranking.executive_summary.system": {
        "title": "Executive Summary System Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "System prompt for generating executive summary (dimension summaries, key points, red flags).",
        "type": "text",
        "required_placeholders": [],
        "default_value": EXECUTIVE_SUMMARY_SYSTEM,
    },
    "ranking.executive_summary.user": {
        "title": "Executive Summary User Prompt",
        "stage": "ranking",
        "category": "Ranking",
        "source_path": "src/agent/prompts/ranking.py",
        "description": "User prompt template for executive summary.",
        "type": "text",
        "required_placeholders": [
            "{company_summary}",
            "{vc_context}",
            "{dimension_block}",
            "{pro_arguments}",
            "{contra_arguments}",
            "{critical_gaps}",
        ],
        "default_value": EXECUTIVE_SUMMARY_USER,
    },
}


def get_default_values() -> dict[str, Any]:
    """Return default prompt values by ID."""
    return {
        prompt_id: deepcopy(PROMPT_DEFINITIONS[prompt_id]["default_value"])
        for prompt_id in ORDERED_PROMPT_IDS
    }


def build_default_catalog() -> dict[str, Any]:
    """Build the default prompt catalog payload for API/UI use."""
    values = get_default_values()
    items: list[dict[str, Any]] = []
    for prompt_id in ORDERED_PROMPT_IDS:
        meta = PROMPT_DEFINITIONS[prompt_id]
        items.append(
            {
                "id": prompt_id,
                "title": meta["title"],
                "stage": meta["stage"],
                "category": meta["category"],
                "source_path": meta["source_path"],
                "description": meta["description"],
                "type": meta["type"],
                "required_placeholders": list(meta["required_placeholders"]),
                "default_value": deepcopy(meta["default_value"]),
                "value": deepcopy(values[prompt_id]),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "items": items}
