"""Answer leaf-level questions using web search.

Leaf nodes in the question tree don't have sub-questions, so they
require external research (web search) to find answers.

This stage uses a custom IntelligentWebSearchTool optimized for
VC investment research with proper rate limiting and error handling.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.common.llm_config import get_llm
from agent.dataclasses.company import Company
from agent.evidence_answering import _coerce_text
from agent.pipeline.state.answer import AnswerState
from agent.pipeline.utils.helpers import generate_context_block
from agent.run_context import use_stage_context
from agent.web_search import get_provider

load_dotenv()


SYSTEM_PROMPT = """
Answer the question using company summary and sub Q&A if provided. Keep answer concise (<50 words) with data backing.
If unable to answer the question, use web_search for market data, trends, competitive analysis, funding info. Focus on industry-level searches, not specific companies. Use the tool only if necessary.
Make ONE tool call at a time.
"""

QUESTION_PROMPT = """
Question: {question}

Company summary: {company_summary}
{context_block}
"""


class IntelligentWebSearchTool(BaseTool):
    """Web search optimized for VC investment research.

    Features:
    - Supports multiple search providers (Sonar, etc.)
    - Configurable search end date for backtesting
    - Robust error handling with helpful messages
    """

    name: str = "web_search"
    description: str = """Search for market data, industry trends, competitive analysis, funding info, and business intelligence.

Query format: Use natural language with key terms. Include:
- Core topic (e.g., "AI business automation market")
- Specifics (e.g., "size", "growth", "forecast", "trends")

Good: "AI business automation market size forecast"

Avoid: repetition, excessive keywords, quotation marks unless needed for exact phrases."""

    def __init__(self, search_end_date: str, **data):
        provider_name = data.pop("provider_name", None)
        super().__init__(**data)
        self._search_end_date = search_end_date
        if not provider_name:
            provider_name = os.getenv("WEB_SEARCH_PROVIDER", "sonar")
        self._provider = get_provider(
            search_end_date=self._search_end_date, provider_name=provider_name
        )

    def _run(self, query: str) -> str:
        """Execute the search with the provided query."""
        try:
            if not query or len(query.strip()) < 2:
                return "Search query too short. Please provide a more specific query."

            raw_results = self._provider.search(query)

            if not raw_results or raw_results.strip() == "":
                return "No search results found. Consider using broader industry terms rather than specific company names."

            if raw_results.startswith("Search Results for:"):
                return raw_results
            return f"Search Results for: {query}\n\n{raw_results}"

        except Exception as e:
            error_msg = str(e)
            if "422" in error_msg:
                return "Search failed due to invalid parameters. Please try a simpler query with broader industry terms."
            elif "401" in error_msg or "403" in error_msg:
                return "Search failed due to authentication issues. Please check API key configuration."
            elif "429" in error_msg:
                return "Search rate limit exceeded. Please try again later."
            else:
                return f"Search failed: {error_msg}. Try broader industry terms rather than specific company names."


def _create_tools_for_state(state: AnswerState) -> list:
    """Create tools dynamically based on the current state."""
    web_search_tool = IntelligentWebSearchTool(search_end_date=state.search_end_date)
    return [web_search_tool]


def limited_tools_condition(state: AnswerState):
    """Route based on tool usage limit.

    Checks if the LLM wants to use tools and if we haven't exceeded
    the tool usage limit. Enforces single tool call at a time.
    """
    if state.tool_usage_count >= state.max_tool_usage:
        if state.answer:
            return END
        return "answer_question"

    tools_decision = tools_condition(state)

    if tools_decision == "tools":
        last_message = state.messages[-1] if state.messages else None
        if (
            last_message
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            num_requested_calls = len(last_message.tool_calls)

            # Keep only the first tool call
            if num_requested_calls > 1:
                last_message.tool_calls = [last_message.tool_calls[0]]

            if state.tool_usage_count + 1 > state.max_tool_usage:
                return "answer_question"

    return tools_decision


def track_tool_usage(state: AnswerState) -> AnswerState:
    """Increment the tool usage counter after tools are used."""
    state.tool_usage_count += 1
    return state


def answer_question(state: AnswerState) -> AnswerState:
    """Generate an answer, optionally using web search.

    If tool usage limit is reached, forces a final answer
    based on available information.
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

    if state.tool_usage_count >= state.max_tool_usage:
        state.messages.append(
            SystemMessage(
                content="Tool limit reached. Provide final answer based on available information. Do not request more tools."
            )
        )
        with use_stage_context("answering"):
            llm = get_llm(temperature=0.0)
            response = llm.invoke(state.messages)
    else:
        tools = _create_tools_for_state(state)
        with use_stage_context("answering"):
            llm = get_llm(temperature=0.0)
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(state.messages)

    state.messages.append(response)

    if response.content:
        state.answer = _coerce_text(response.content)

    return state


def dynamic_tool_node(state: AnswerState) -> AnswerState:
    """Execute tools dynamically based on current state."""
    tools = _create_tools_for_state(state)
    tool_node = ToolNode(tools)
    return tool_node.invoke(state)


# Build the graph
builder = StateGraph(AnswerState)

builder.add_node("answer_question", answer_question)
builder.add_node("tools", dynamic_tool_node)
builder.add_node("track_tool_usage", track_tool_usage)

builder.add_edge(START, "answer_question")
builder.add_conditional_edges("answer_question", limited_tools_condition)
builder.add_edge("tools", "track_tool_usage")
builder.add_edge("track_tool_usage", "answer_question")

graph = builder.compile()


if __name__ == "__main__":
    simple_company = Company(
        name="TechStartup AI",
        industry="Artificial Intelligence",
        tagline="AI-powered automation for businesses",
        about="A startup building AI tools for business process automation.",
    )

    initial_state = AnswerState(
        question="What is the market size for AI-powered business automation?",
        company=simple_company,
        is_backtesting=True,
        search_end_date="2021-06-15",
    )

    print("=" * 60)
    print("Running answer_question with example question")
    print("=" * 60)
    print(f"Question: {initial_state.question}")
    print(f"Company: {simple_company.name}")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print(f"\nAnswer: {result['answer']}")
    print(f"Tool usage: {result['tool_usage_count']}/{result['max_tool_usage']}")
