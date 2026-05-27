"""Synthesize answers for parent nodes from child answers.

Parent nodes have sub-questions that are already answered. This stage
combines those answers into a coherent summary without external research.

This is used for non-leaf nodes in the question tree where the answer
can be synthesized from the child Q&A pairs.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.common.llm_config import get_llm
from agent.dataclasses.company import Company
from agent.evidence_answering import _coerce_text
from agent.pipeline.state.answer import AnswerStateSimple
from agent.pipeline.utils.helpers import generate_context_block
from agent.run_context import use_stage_context

SYSTEM_PROMPT = """
Answer using company summary and sub Q&A if provided. Keep answer concise (<50 words) with data backing.
"""

QUESTION_PROMPT = """
Question: {question}

Company summary: {company_summary}
{context_block}
"""

def answer_question(state: AnswerStateSimple) -> AnswerStateSimple:
    """Synthesize answer from child Q&A pairs.

    This generates a concise answer by combining information from
    the sub-question answers without making external tool calls.
    """
    if len(state.messages) == 0:
        state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=QUESTION_PROMPT.format(
                    question=state.question,
                    company_summary=state.company.get_company_summary(),
                    context_block=generate_context_block(
                        state.qa_pairs, state.vc_context
                    ),
                )
            ),
        ]

    with use_stage_context("answering"):
        llm = get_llm(temperature=0.0)
        response = llm.invoke(state.messages)
    state.answer = _coerce_text(response.content)

    return state


# Build the graph
builder = StateGraph(AnswerStateSimple)

builder.add_node("answer_question", answer_question)

builder.add_edge(START, "answer_question")
builder.add_edge("answer_question", END)

graph = builder.compile()


if __name__ == "__main__":
    # Simple test
    simple_company = Company(
        name="TechStartup AI",
        industry="Artificial Intelligence",
        tagline="AI-powered automation for businesses",
        about="A startup building AI tools for business process automation.",
    )

    initial_state = AnswerStateSimple(
        question="What is the market size for AI-powered business automation?",
        company=simple_company,
    )

    result = graph.invoke(initial_state)
    print(f"Answer: {result['answer']}")
