import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.dataclasses.config import Config
from agent.ingest.store import Chunk, EvidenceStore
from agent.llm_policy import build_phase_model_policy, build_pipeline_policy
from agent.pipeline.state.decomposition import DecompositionInput, DecompositionNode, DecompositionTree
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.state.schemas import (
    ArgumentCritique,
    ArgumentsOutput,
    CriterionScore,
    DimensionScoreOutput,
    ExecutiveSummaryOutput,
    IndividualRefinedArgumentOutput,
    SingleArgumentScore,
)
from agent.pipeline.stages import critique, decomposition, evaluation, generation, ranking, refinement
from agent.run_context import get_current_llm_selection, get_current_stage_name, use_run_context
import agent.evidence_answering as evidence_answering


class _FakeRunnable:
    def __init__(self, result, sink):
        self._result = result
        self._sink = sink

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        selection = get_current_llm_selection() or {}
        self._sink.append((selection.get("provider"), selection.get("model")))
        return self._result

    async def ainvoke(self, _messages):
        selection = get_current_llm_selection() or {}
        self._sink.append((selection.get("provider"), selection.get("model")))
        return self._result


class _AuthFallbackRunnable:
    def __init__(self, result, sink, failing_provider):
        self._result = result
        self._sink = sink
        self._failing_provider = failing_provider

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        selection = get_current_llm_selection() or {}
        current = (selection.get("provider"), selection.get("model"))
        self._sink.append(current)
        if current[0] == self._failing_provider:
            raise Exception("Error code: 401 - {'type':'error','error':{'type':'authentication_error','message':'invalid x-api-key'}}")
        return self._result

    async def ainvoke(self, _messages):
        selection = get_current_llm_selection() or {}
        current = (selection.get("provider"), selection.get("model"))
        self._sink.append(current)
        if current[0] == self._failing_provider:
            raise Exception("Error code: 401 - {'type':'error','error':{'type':'authentication_error','message':'invalid x-api-key'}}")
        return self._result


class _StageTrackingRunnable:
    def __init__(self, result, sink):
        self._result = result
        self._sink = sink

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        self._sink.append(get_current_stage_name())
        return self._result

    async def ainvoke(self, _messages):
        self._sink.append(get_current_stage_name())
        return self._result


class _RankingSelectionRunnable:
    def __init__(self, sink):
        self._sink = sink

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        selection = get_current_llm_selection() or {}
        self._sink.append((selection.get("provider"), selection.get("model")))
        if get_current_stage_name() == "ranking_executive_summary":
            return ExecutiveSummaryOutput(
                strategy_fit_summary="fit",
                team_summary="team",
                potential_summary="potential",
                key_points=["k1"],
                red_flags=["r1"],
            )
        return DimensionScoreOutput(
            raw_score=80,
            confidence=0.8,
            evidence_count=1,
            top_qa_indices=[0],
            evidence_snippets=["evidence"],
            critical_gaps=[],
        )


class _RankingRunnable:
    def __init__(self, sink):
        self._sink = sink

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        stage = get_current_stage_name()
        self._sink.append(stage)
        if stage == "ranking_executive_summary":
            return ExecutiveSummaryOutput(
                strategy_fit_summary="fit",
                team_summary="team",
                potential_summary="potential",
                key_points=["k1"],
                red_flags=["r1"],
            )
        return DimensionScoreOutput(
            raw_score=80,
            confidence=0.8,
            evidence_count=1,
            top_qa_indices=[0],
            evidence_snippets=["evidence"],
            critical_gaps=[],
        )


def test_pipeline_stages_use_phase_policy(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy(
        "premium",
        {
            "decomposition": "claude",
            "generation": "gpt5",
            "evaluation": "claude",
            "ranking": "gpt5",
        },
    )
    company = Company(name="Acme", industry="Fintech")
    qa_pairs = [{"question": "Q?", "answer": "A", "aspect": "general_company"}]
    state = IterativeInvestmentStoryState(
        company=company,
        config=Config(
            n_pro_arguments=1,
            n_contra_arguments=1,
            k_best_arguments_per_iteration=[1],
            max_iterations=1,
        ),
        all_qa_pairs=qa_pairs,
        current_arguments=[Argument(content="Argument", argument_type="pro", qa_indices=[0], critique="Need more support")],
        selected_arguments=[Argument(content="Argument", argument_type="pro", qa_indices=[0], argument_feedback="Feedback")],
        final_arguments=[Argument(content="Final pro", argument_type="pro", qa_indices=[0], score=10)],
    )

    seen = {
        "decomposition": [],
        "answering": [],
        "generation": [],
        "critique": [],
        "evaluation": [],
        "refinement": [],
        "ranking": [],
    }

    monkeypatch.setattr(
        decomposition,
        "get_llm",
        lambda temperature=0.0: _FakeRunnable(
            DecompositionTree(nodes=[DecompositionNode(question="Root?", sub_questions=[])]),
            seen["decomposition"],
        ),
    )
    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda temperature=0.0: _FakeRunnable(
            ArgumentsOutput(arguments=[{"content": "Pro case", "qa_indices": [0]}]),
            seen["generation"],
        ),
    )
    monkeypatch.setattr(
        critique,
        "get_llm",
        lambda temperature=0.0: _FakeRunnable(
            ArgumentCritique(critique="Counterpoint"),
            seen["critique"],
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "get_llm",
        lambda temperature=0.0: _FakeRunnable(
            SingleArgumentScore(scores=[CriterionScore(score=1, reasoning="ok") for _ in range(14)]),
            seen["evaluation"],
        ),
    )
    monkeypatch.setattr(
        refinement,
        "get_llm",
        lambda temperature=0.0: _FakeRunnable(
            IndividualRefinedArgumentOutput(content="Refined", qa_indices=[0]),
            seen["refinement"],
        ),
    )
    monkeypatch.setattr(
        ranking,
        "get_llm",
        lambda temperature=0.0, reasoning_effort=None: _RankingSelectionRunnable(seen["ranking"]),
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        decomposition.decompose_question(
            DecompositionInput(question="Root?", industry="Fintech", aspect="general_company")
        )
        generation.generate_pro_arguments(state)
        asyncio.run(
            critique._apply_devils_advocate_to_pro_argument(
                Argument(content="Argument", argument_type="pro", qa_indices=[0]),
                "Q/A",
            )
        )
        asyncio.run(
            evaluation.score_single_argument(
                Argument(content="Argument", argument_type="pro", qa_indices=[0], critique="Need more support")
            )
        )
        asyncio.run(
            refinement._refine_individual_pro_argument(
                Argument(content="Argument", argument_type="pro", qa_indices=[0], argument_feedback="Feedback"),
                "Q/A",
            )
        )
        ranked = ranking.score_company_dimensions(state)["ranking_result"]
        state.ranking_result = ranked
        ranking.generate_executive_summary(state)

    assert seen["decomposition"][-1] == ("anthropic", "claude-haiku-4-5-20251001")
    assert seen["generation"][-1] == ("openai", "gpt-5")
    assert seen["critique"][-1] == ("gemini", "gemini-3.1-flash-lite-preview")
    assert seen["evaluation"][-1] == ("anthropic", "claude-haiku-4-5-20251001")
    assert seen["refinement"][-1] == ("gemini", "gemini-3.1-flash-lite-preview")
    assert all(item == ("openai", "gpt-5") for item in seen["ranking"])


def test_pipeline_stages_set_stage_context_for_critical_llm_calls(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {})
    company = Company(name="Acme", industry="Fintech")
    qa_pairs = [{"question": "Q?", "answer": "A", "aspect": "general_company"}]
    state = IterativeInvestmentStoryState(
        company=company,
        config=Config(
            n_pro_arguments=1,
            n_contra_arguments=1,
            k_best_arguments_per_iteration=[1],
            max_iterations=1,
        ),
        all_qa_pairs=qa_pairs,
        current_arguments=[
            Argument(
                content="Argument",
                argument_type="pro",
                qa_indices=[0],
                critique="Need more support",
            )
        ],
        final_arguments=[Argument(content="Final pro", argument_type="pro", qa_indices=[0], score=10)],
    )

    seen = {
        "decomposition": [],
        "generation": [],
        "evaluation": [],
        "ranking": [],
    }

    monkeypatch.setattr(
        decomposition,
        "get_llm",
        lambda temperature=0.0: _StageTrackingRunnable(
            DecompositionTree(nodes=[DecompositionNode(question="Root?", sub_questions=[])]),
            seen["decomposition"],
        ),
    )
    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda temperature=0.0: _StageTrackingRunnable(
            ArgumentsOutput(arguments=[{"content": "Pro case", "qa_indices": [0]}]),
            seen["generation"],
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "get_llm",
        lambda temperature=0.0: _StageTrackingRunnable(
            SingleArgumentScore(scores=[CriterionScore(score=1, reasoning="ok") for _ in range(14)]),
            seen["evaluation"],
        ),
    )
    monkeypatch.setattr(
        ranking,
        "get_llm",
        lambda temperature=0.0, reasoning_effort=None: _RankingRunnable(seen["ranking"]),
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        decomposition.decompose_question(
            DecompositionInput(question="Root?", industry="Fintech", aspect="general_company")
        )
        generation.generate_pro_arguments(state)
        asyncio.run(
            evaluation.score_single_argument(
                Argument(
                    content="Argument",
                    argument_type="pro",
                    qa_indices=[0],
                    critique="Need more support",
                )
            )
        )
        ranked = ranking.score_company_dimensions(state)["ranking_result"]
        state.ranking_result = ranked
        ranking.generate_executive_summary(state)

    assert seen["decomposition"] == ["decomposition"]
    assert seen["generation"] == ["generation_pro"]
    assert seen["evaluation"] == ["evaluation"]
    assert seen["ranking"] == [
        "ranking_dimension_score",
        "ranking_dimension_score",
        "ranking_upside_score",
        "ranking_executive_summary",
    ]


def test_evidence_answering_uses_answering_policy(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {})
    seen: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(
        evidence_answering,
        "create_llm",
        lambda temperature=0.0: _FakeRunnable(type("Resp", (), {"content": "Grounded answer"})(), seen),
    )
    monkeypatch.setattr(
        evidence_answering,
        "retrieve_chunks",
        lambda question, store, k=8: [
            Chunk(
                chunk_id="chunk_1",
                text="Evidence text",
                source_file="deck.txt",
                page_or_slide="1",
            )
        ],
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        answer, _ = asyncio.run(
            evidence_answering.answer_question_from_evidence(
                "What is traction?",
                Company(name="Acme", industry="Fintech"),
                EvidenceStore(startup_slug="acme", chunks=[]),
                use_web_search=False,
            )
        )

    assert answer == "Grounded answer"
    assert seen[-1] == ("gemini", "gemini-3.1-flash-lite-preview")


def test_ranking_falls_back_from_claude_to_gpt5_on_auth_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {"ranking": "claude"})
    seen: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(
        ranking,
        "get_llm",
        lambda temperature=0.0, reasoning_effort=None: _AuthFallbackRunnable(
            DimensionScoreOutput(
                raw_score=82,
                confidence=0.9,
                evidence_count=1,
                top_qa_indices=[0],
                evidence_snippets=["evidence"],
                critical_gaps=[],
            ),
            seen,
            failing_provider="anthropic",
        ),
    )

    company = Company(name="Acme", industry="Fintech")
    state = IterativeInvestmentStoryState(
        company=company,
        config=Config(
            n_pro_arguments=1,
            n_contra_arguments=1,
            k_best_arguments_per_iteration=[1],
            max_iterations=1,
        ),
        all_qa_pairs=[{"question": "Q?", "answer": "A", "aspect": "general_company"}],
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        result = ranking.score_company_dimensions(state)["ranking_result"]

    assert result.strategy_fit_score > 0
    assert result.dimension_scores[0].top_qa_indices == [0]
    assert seen == [
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("openai", "gpt-5"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("openai", "gpt-5"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("openai", "gpt-5"),
    ]


def test_evaluation_falls_back_from_gpt5_to_claude_on_auth_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {"evaluation": "gpt5"})
    seen: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(
        evaluation,
        "get_llm",
        lambda temperature=0.0: _AuthFallbackRunnable(
            SingleArgumentScore(
                scores=[CriterionScore(score=1, reasoning="ok") for _ in range(14)]
            ),
            seen,
            failing_provider="openai",
        ),
    )

    argument = Argument(
        content="Argument",
        argument_type="pro",
        qa_indices=[0],
        critique="Need more support",
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        scored = asyncio.run(evaluation.score_single_argument(argument))

    assert scored.score == 14
    assert seen == [
        ("openai", "gpt-5"),
        ("anthropic", "claude-haiku-4-5-20251001"),
    ]


def test_ranking_keeps_only_valid_top_qa_indices_for_dimension(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {"ranking": "gpt5"})

    monkeypatch.setattr(
        ranking,
        "get_llm",
        lambda temperature=0.0, reasoning_effort=None: _FakeRunnable(
            DimensionScoreOutput(
                raw_score=82,
                confidence=0.9,
                evidence_count=2,
                top_qa_indices=[2, 99, 2, 0],
                evidence_snippets=["evidence"],
                critical_gaps=[],
            ),
            [],
        ),
    )

    company = Company(name="Acme", industry="Fintech")
    state = IterativeInvestmentStoryState(
        company=company,
        config=Config(
            n_pro_arguments=1,
            n_contra_arguments=1,
            k_best_arguments_per_iteration=[1],
            max_iterations=1,
        ),
        all_qa_pairs=[
            {"question": "Stage?", "answer": "Seed", "aspect": "general_company"},
            {"question": "Founder?", "answer": "Repeat founder", "aspect": "team"},
            {"question": "TAM?", "answer": "$2B", "aspect": "market"},
        ],
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        result = ranking.score_company_dimensions(state)["ranking_result"]

    by_dimension = {score.dimension: score for score in result.dimension_scores}
    assert by_dimension["strategy_fit"].top_qa_indices == [0]
    assert by_dimension["team"].top_qa_indices == []
    assert by_dimension["upside"].top_qa_indices == [2]


def test_ranking_uses_raw_upside_score_and_potential_specific_stage(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_pipeline_policy("premium", {"ranking": "gpt5"})

    seen_stages: list[str] = []
    seen_temperatures: list[tuple[str | None, float | None]] = []

    class _PerDimensionRunnable:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _messages):
            stage = get_current_stage_name()
            seen_stages.append(stage)
            if stage == "ranking_upside_score":
                seen_temperatures.append((stage, 0.7))
                return DimensionScoreOutput(
                    raw_score=95,
                    confidence=0.1,
                    evidence_count=1,
                    top_qa_indices=[0],
                    evidence_snippets=["huge upside"],
                    critical_gaps=["execution risk"],
                )
            seen_temperatures.append((stage, 0.0))
            return DimensionScoreOutput(
                raw_score=80,
                confidence=0.5,
                evidence_count=1,
                top_qa_indices=[0],
                evidence_snippets=["evidence"],
                critical_gaps=[],
            )

    monkeypatch.setattr(
        ranking,
        "get_llm",
        lambda temperature=0.0, reasoning_effort=None: _PerDimensionRunnable(),
    )

    company = Company(name="Acme", industry="Fintech")
    state = IterativeInvestmentStoryState(
        company=company,
        config=Config(
            n_pro_arguments=1,
            n_contra_arguments=1,
            k_best_arguments_per_iteration=[1],
            max_iterations=1,
        ),
        all_qa_pairs=[
            {"question": "Stage?", "answer": "Seed", "aspect": "general_company"},
            {"question": "Founder?", "answer": "Repeat founder", "aspect": "team"},
            {"question": "TAM?", "answer": "$5B", "aspect": "market"},
        ],
    )

    with use_run_context(llm_selection=policy.answering, pipeline_policy=policy):
        result = ranking.score_company_dimensions(state)["ranking_result"]

    assert result.strategy_fit_score == 68.0
    assert result.team_score == 68.0
    assert result.upside_score == 95
    assert seen_stages == [
        "ranking_dimension_score",
        "ranking_dimension_score",
        "ranking_upside_score",
    ]
    assert seen_temperatures == [
        ("ranking_dimension_score", 0.0),
        ("ranking_dimension_score", 0.0),
        ("ranking_upside_score", 0.7),
    ]


def test_pipeline_policy_can_route_five_user_selected_models(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    policy = build_phase_model_policy(
        {
            "decomposition": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
            "generation": {"provider": "openai", "model": "gpt-5"},
            "evaluation": {"provider": "openai", "model": "gpt-5.4-mini"},
            "ranking": {"provider": "openai", "model": "gpt-4.1-mini"},
        }
    )

    assert policy.decomposition["model"] == "claude-haiku-4-5-20251001"
    assert policy.answering["model"] == "gemini-3.1-flash-lite-preview"
    assert policy.critique["model"] == "gemini-3.1-flash-lite-preview"
    assert policy.refinement["model"] == "gemini-3.1-flash-lite-preview"
    assert policy.generation["model"] == "gpt-5"
    assert policy.evaluation["model"] == "gpt-5.4-mini"
    assert policy.ranking["model"] == "gpt-4.1-mini"
