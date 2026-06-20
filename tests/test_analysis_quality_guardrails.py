import asyncio
from contextlib import contextmanager
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


_STUBBED_MODULE_NAMES = (
    "supabase",
    "agent",
    "agent.llm_catalog",
    "agent.llm_policy",
    "agent.company_chat",
    "agent.pipeline",
    "agent.pipeline.scoring_signals",
    "agent.run_context",
    "agent.person_intel",
    "agent.person_intel.models",
    "agent.person_intel.service",
    "agent.dataclasses",
    "agent.dataclasses.company",
    "agent.ingest",
    "agent.ingest.store",
    "agent.ingest.chunking",
    "agent.ingest.specter_augmentation",
    "agent.ingest.specter_ingest",
    "agent.ingest.specter_mcp_client",
)


def _install_lightweight_agent_stubs() -> None:
    """Avoid importing the full analysis pipeline for app preflight tests."""

    from pydantic import BaseModel, Field

    supabase_module = types.ModuleType("supabase")
    supabase_module.Client = type("Client", (), {})
    supabase_module.create_client = lambda *_args, **_kwargs: supabase_module.Client()
    sys.modules.setdefault("supabase", supabase_module)

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = [str(ROOT / "src" / "agent")]
    sys.modules.setdefault("agent", agent_pkg)

    llm_catalog = types.ModuleType("agent.llm_catalog")
    llm_catalog.available_models_payload = lambda: []
    llm_catalog.available_chat_models_payload = lambda: []
    llm_catalog.model_label = lambda provider=None, model=None: model or "Test Model"
    llm_catalog.current_default_selection = lambda: {"provider": "openai", "model": "test-model"}
    llm_catalog.normalize_creativity = lambda value=None: value or "balanced"
    llm_catalog.normalize_provider = lambda value=None: value
    llm_catalog.pricing_catalog_payload = lambda: {}
    llm_catalog.serialize_selection = (
        lambda provider=None, model=None: {
            "provider": provider or "openai",
            "model": model or "test-model",
            "label": model or "Test Model",
        }
    )
    llm_catalog.validate_chat_requested_selection = (
        lambda provider=None, model=None: SimpleNamespace(
            provider=provider or "openai",
            model=model or "test-model",
        )
    )
    llm_catalog.validate_requested_selection = llm_catalog.validate_chat_requested_selection
    sys.modules.setdefault("agent.llm_catalog", llm_catalog)

    llm_policy = types.ModuleType("agent.llm_policy")
    llm_policy.PHASE_LABELS = {}
    llm_policy.PHASE_SHORT_LABELS = {}
    llm_policy.build_default_phase_model_policy = lambda: SimpleNamespace(
        answering={"provider": "openai", "model": "test-model", "label": "Test Model"}
    )
    llm_policy.build_phase_model_policy = llm_policy.build_default_phase_model_policy
    llm_policy.build_phase_policy_display_label = lambda *_args, **_kwargs: "Test Model"
    llm_policy.build_pipeline_policy = lambda *_args, **_kwargs: llm_policy.build_default_phase_model_policy()
    llm_policy.build_tier_display_label = lambda *_args, **_kwargs: "Test Model"
    llm_policy.coerce_phase_models_payload = lambda value=None: value or {}
    llm_policy.default_phase_model_selections = lambda: {}
    llm_policy.normalize_phase_models = lambda value=None: value or {}
    llm_policy.normalize_premium_phase_models = lambda value=None: value or {}
    llm_policy.normalize_quality_tier = lambda value=None: value
    llm_policy.phase_model_defaults_payload = lambda: {}
    llm_policy.premium_phase_options_payload = lambda: {}
    llm_policy.quality_tiers_payload = lambda: {}
    llm_policy.resolve_effective_phase_choices = lambda *_args, **_kwargs: {}
    llm_policy.resolve_effective_phase_models = lambda *_args, **_kwargs: {}
    sys.modules.setdefault("agent.llm_policy", llm_policy)

    company_chat = types.ModuleType("agent.company_chat")
    company_chat.answer_company_question = lambda *_args, **_kwargs: {}
    company_chat.build_company_chat_store = lambda *_args, **_kwargs: (None, None, {}, {})
    sys.modules.setdefault("agent.company_chat", company_chat)

    scoring_signals = types.ModuleType("agent.pipeline.scoring_signals")
    scoring_signals.build_scoring_signals = lambda *_args, **_kwargs: {}
    sys.modules.setdefault("agent.pipeline", types.ModuleType("agent.pipeline"))
    sys.modules.setdefault("agent.pipeline.scoring_signals", scoring_signals)

    @contextmanager
    def null_context(*_args, **_kwargs):
        yield None

    class DummyTelemetryCollector:
        def __init__(self, *args, **kwargs):
            self.model_executions = []

    run_context = types.ModuleType("agent.run_context")
    run_context.RunTelemetryCollector = DummyTelemetryCollector
    run_context.build_run_costs_from_model_executions = lambda *_args, **_kwargs: {}
    run_context.use_company_context = null_context
    run_context.use_run_context = null_context
    sys.modules.setdefault("agent.run_context", run_context)

    class PersonProfileJobRequest(BaseModel):
        primary_profile_url: str | None = None
        full_name: str | None = None
        location: str | None = None
        current_company: str | None = None
        role: str | None = None
        known_aliases: list[str] = Field(default_factory=list)
        user_uploaded_text: str | None = None
        user_uploaded_images: list[str] = Field(default_factory=list)
        company_slug: str | None = None
        person_key: str | None = None

    class FounderRequest(PersonProfileJobRequest):
        pass

    class BulkFounderJobRequest(BaseModel):
        company_slug: str = ""
        founders: list[FounderRequest] = Field(default_factory=list)
        user_uploaded_text: str | None = None
        user_uploaded_images: list[str] = Field(default_factory=list)

    person_models = types.ModuleType("agent.person_intel.models")
    person_models.BulkFounderJobRequest = BulkFounderJobRequest
    person_models.PersonProfileJobRequest = PersonProfileJobRequest
    sys.modules.setdefault("agent.person_intel", types.ModuleType("agent.person_intel"))
    sys.modules.setdefault("agent.person_intel.models", person_models)

    class DummyPersonIntelService:
        async def build_profile(self, *_args, **_kwargs):
            profile = SimpleNamespace(claims=[], model_dump=lambda: {})
            return profile, "", "cache-key", {}

    person_service = types.ModuleType("agent.person_intel.service")
    person_service.PersonIntelService = DummyPersonIntelService
    sys.modules.setdefault("agent.person_intel.service", person_service)

    class Company:
        def __init__(self, name=None, domain=None, company_url=None, **kwargs):
            self.name = name
            self.domain = domain
            self.company_url = company_url
            self.industry = kwargs.get("industry")
            self.tagline = kwargs.get("tagline")
            self.about = kwargs.get("about")
            self.team = kwargs.get("team")

    dataclasses_pkg = types.ModuleType("agent.dataclasses")
    company_module = types.ModuleType("agent.dataclasses.company")
    company_module.Company = Company
    sys.modules.setdefault("agent.dataclasses", dataclasses_pkg)
    sys.modules.setdefault("agent.dataclasses.company", company_module)

    class Chunk(SimpleNamespace):
        pass

    class EvidenceStore:
        def __init__(self, startup_slug: str, chunks=None):
            self.startup_slug = startup_slug
            self.chunks = list(chunks or [])

    def smart_chunk_texts(raw_items):
        chunks = []
        for idx, item in enumerate(raw_items, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{idx}",
                    text=item.get("text", ""),
                    source_file=item.get("source_file"),
                    page_or_slide=item.get("page_or_slide"),
                )
            )
        return chunks

    ingest_module = types.ModuleType("agent.ingest")
    ingest_module._EXTENSION_MAP = {}
    ingest_module._TEXT_EXTENSIONS = {".txt", ".md"}
    ingest_module.EvidenceStore = EvidenceStore
    ingest_module.ingest_startup_folder = lambda *_args, **_kwargs: EvidenceStore("startup")
    ingest_store = types.ModuleType("agent.ingest.store")
    ingest_store.Chunk = Chunk
    ingest_store.EvidenceStore = EvidenceStore
    chunking_module = types.ModuleType("agent.ingest.chunking")
    chunking_module.smart_chunk_texts = smart_chunk_texts
    specter_aug = types.ModuleType("agent.ingest.specter_augmentation")
    specter_aug.extract_company_url = lambda *_args, **_kwargs: None
    specter_ingest = types.ModuleType("agent.ingest.specter_ingest")
    specter_ingest.list_specter_companies = lambda *_args, **_kwargs: []
    specter_ingest.ingest_specter = lambda *_args, **_kwargs: []
    specter_ingest.ingest_specter_company = lambda *_args, **_kwargs: (None, EvidenceStore("startup"))
    specter_mcp = types.ModuleType("agent.ingest.specter_mcp_client")
    specter_mcp.fetch_specter_company = lambda *_args, **_kwargs: (None, None)
    sys.modules.setdefault("agent.ingest", ingest_module)
    sys.modules.setdefault("agent.ingest.store", ingest_store)
    sys.modules.setdefault("agent.ingest.chunking", chunking_module)
    sys.modules.setdefault("agent.ingest.specter_augmentation", specter_aug)
    sys.modules.setdefault("agent.ingest.specter_ingest", specter_ingest)
    sys.modules.setdefault("agent.ingest.specter_mcp_client", specter_mcp)


@pytest.fixture(scope="module")
def guardrail_modules():
    sentinel = object()
    saved_modules = {
        name: sys.modules.get(name, sentinel)
        for name in (*_STUBBED_MODULE_NAMES, "web.app", "web.db")
    }
    web_package = sys.modules.get("web")
    had_web_app_attr = web_package is not None and hasattr(web_package, "app")
    had_web_db_attr = web_package is not None and hasattr(web_package, "db")
    saved_web_app_attr = getattr(web_package, "app", sentinel) if web_package is not None else sentinel
    saved_web_db_attr = getattr(web_package, "db", sentinel) if web_package is not None else sentinel

    _install_lightweight_agent_stubs()
    web_app = importlib.import_module("web.app")
    web_db = importlib.import_module("web.db")
    try:
        yield SimpleNamespace(app=web_app, db=web_db)
    finally:
        for name, module in saved_modules.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        web_package = sys.modules.get("web")
        if web_package is not None:
            if had_web_app_attr:
                setattr(web_package, "app", saved_web_app_attr)
            elif hasattr(web_package, "app"):
                delattr(web_package, "app")
            if had_web_db_attr:
                setattr(web_package, "db", saved_web_db_attr)
            elif hasattr(web_package, "db"):
                delattr(web_package, "db")

from web.analysis_quality import normalize_company_display_name


def _reset_app_state(monkeypatch, web_app) -> None:
    web_app._jobs.clear()
    web_app._job_controls.clear()
    web_app._results_cache.clear()
    web_app._jobs_overview_cache.update({"expires_at": 0.0, "payload": None})
    web_app._company_runs_cache.update({"expires_at": 0.0, "payload": None})
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: False))
    # _start_analysis_job resolves a default phase-model policy, which needs at
    # least one provider key; CI has none and tests must not rely on a local .env.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _json_response_body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_company_display_name_normalizes_hash_prefixed_pitchdeck_slug() -> None:
    normalized = normalize_company_display_name("a22e3360c47b-varieneai-pitch-deck")

    assert normalized.valid is True
    assert normalized.value == "VarieneAI"


def test_company_display_name_rejects_domain_only_csv_name() -> None:
    normalized = normalize_company_display_name("variene.ai", source="specter_csv")

    assert normalized.valid is False
    assert "domain" in normalized.reason.lower()


def test_company_display_name_preserves_dotted_brand_casing() -> None:
    normalized = normalize_company_display_name("f.nous-pitch-deck.pdf")

    assert normalized.valid is True
    assert normalized.value == "F.NOUS"


def test_preflight_blocks_empty_specter_run_before_start(monkeypatch, tmp_path: Path, guardrail_modules) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    job_id = "0ddb61a7"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [],
        "specter": {},
        "specter_urls": [],
    }

    def fail_thread(*args, **kwargs):
        raise AssertionError("analysis thread must not start")

    monkeypatch.setattr(web_app.threading, "Thread", fail_thread)

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="specter"),
            started_by={},
            require_identity=False,
        )
    )

    body = _json_response_body(response)
    assert response.status_code == 409
    assert body["code"] == "quality_preflight_failed"
    assert "zero usable company inputs" in body["invalid_inputs"][-1]["reason"]
    assert web_app._jobs[job_id].status == "pending"


def test_preflight_blocks_unresolved_specter_url_before_worker_queue(
    monkeypatch,
    tmp_path: Path,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    job_id = "5199998a"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [],
        "specter": {},
        "specter_urls": ["variene.ai"],
    }
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(
        web_app,
        "_resolve_specter_url_for_preflight",
        lambda item: asyncio.sleep(0, result=(None, "Specter company worker exited with code 1")),
    )
    monkeypatch.setattr(
        web_app,
        "_queue_worker_backed_specter_job",
        lambda current_job_id: (_ for _ in ()).throw(AssertionError("worker queue must not be called")),
    )

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="specter"),
            started_by={},
            require_identity=False,
        )
    )

    body = _json_response_body(response)
    assert response.status_code == 409
    assert body["invalid_inputs"][0]["input"] == "variene.ai"
    assert "could not resolve" in body["invalid_inputs"][0]["reason"]
    assert web_app._jobs[job_id].progress == "Analysis blocked by quality preflight."
    assert any(
        "Specter company worker exited with code 1" in message
        for message in web_app._jobs[job_id].progress_log
    )


def test_persisted_preflight_status_is_added_to_progress_log(
    monkeypatch,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_analysis_events(job_id, limit):
            assert job_id == "blocked1"
            assert limit == web_app.MAX_PROGRESS_LOG_ENTRIES
            return ["Received 1 URLs for Specter MCP intake"]

        @staticmethod
        def load_job_status(job_id):
            assert job_id == "blocked1"
            return {
                "status": "pending",
                "progress": "Analysis blocked by quality preflight.",
                "worker_active": False,
            }

    monkeypatch.setattr(web_app, "db", FakeDb)

    assert web_app._load_worker_progress_log("blocked1") == [
        "Received 1 URLs for Specter MCP intake",
        "Analysis blocked by quality preflight.",
    ]


def test_preflight_allows_resolved_specter_url_and_queues_worker(
    monkeypatch,
    tmp_path: Path,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    queued = {"called": False}
    job_id = "validurl1"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [],
        "specter": {},
        "specter_urls": ["https://variene.ai"],
    }
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(
        web_app,
        "_resolve_specter_url_for_preflight",
        lambda item: asyncio.sleep(
            0,
            result=(SimpleNamespace(name="VarieneAI", domain="variene.ai"), None),
        ),
    )

    def queue_worker(current_job_id):
        queued["called"] = current_job_id == job_id
        return True, "Queued for worker."

    monkeypatch.setattr(web_app, "_queue_worker_backed_specter_job", queue_worker)

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="specter"),
            started_by={},
            require_identity=False,
        )
    )

    assert response["status"] == "running"
    assert queued["called"] is True
    assert web_app._results_cache[job_id]["quality_preflight"]["status"] == "ready"


def test_preflight_allows_valid_specter_csv_and_starts_thread(
    monkeypatch,
    tmp_path: Path,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    started = {"called": False}
    job_id = "validcsv1"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    companies_csv = upload_dir / "companies.csv"
    companies_csv.write_text("Company Name,Domain\nVarieneAI,variene.ai\n", encoding="utf-8")
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [{"name": companies_csv.name, "local_path": str(companies_csv), "size": companies_csv.stat().st_size}],
        "specter": {"companies": str(companies_csv)},
        "specter_urls": [],
    }
    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda path: [{"name": "VarieneAI", "slug": "varieneai", "domain": "variene.ai"}],
    )

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            started["called"] = True

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="specter"),
            started_by={},
            require_identity=False,
        )
    )

    assert response["status"] == "running"
    assert started["called"] is True
    assert web_app._results_cache[job_id]["quality_preflight"]["status"] == "ready"


def test_preflight_allows_hash_prefixed_pitchdeck_after_normalization(
    monkeypatch,
    tmp_path: Path,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    started = {"called": False}
    job_id = "942b82c0"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    deck = upload_dir / "a22e3360c47b-varieneai-pitch-deck.txt"
    deck.write_text("VarieneAI builds applied AI tools for industrial teams.", encoding="utf-8")
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [{"name": deck.name, "local_path": str(deck), "size": deck.stat().st_size}],
        "specter": None,
    }

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            started["called"] = True

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="pitchdeck", use_specter_mcp=False),
            started_by={},
            require_identity=False,
        )
    )

    identities = web_app._results_cache[job_id]["quality_preflight_document_identities"]
    assert response["status"] == "running"
    assert started["called"] is True
    assert identities[deck.name]["company_name"] == "VarieneAI"


def test_preflight_allows_valid_original_mode_with_run_name(
    monkeypatch,
    tmp_path: Path,
    guardrail_modules,
) -> None:
    web_app = guardrail_modules.app
    _reset_app_state(monkeypatch, web_app)
    started = {"called": False}
    job_id = "original1"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    document = upload_dir / "uploaded-notes.txt"
    # No URL in the document: with a detectable URL the preflight derives the
    # company name from the domain instead of the run_name under test.
    document.write_text("Industrial AI evidence without a website mention.", encoding="utf-8")
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [{"name": document.name, "local_path": str(document), "size": document.stat().st_size}],
        "specter": None,
    }

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            started["called"] = True

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    response = asyncio.run(
        web_app._start_analysis_job(
            job_id,
            web_app.AnalyzeRequest(input_mode="original", run_name="VarieneAI", use_specter_mcp=False),
            started_by={},
            require_identity=False,
        )
    )

    identities = web_app._results_cache[job_id]["quality_preflight_document_identities"]
    assert response["status"] == "running"
    assert started["called"] is True
    assert identities[document.name]["company_name"] == "VarieneAI"


def test_company_runs_extraction_skips_failed_rows_and_normalizes_names(guardrail_modules) -> None:
    web_db = guardrail_modules.db
    failed = web_db._extract_company_runs_from_payload(
        "5199998a",
        {
            "mode": "batch",
            "summary_rows": [
                {"company_name": "variene.ai", "startup_slug": "variene-ai", "decision": "error"}
            ],
        },
        created_at="2026-05-15T07:50:00+00:00",
        mode="specter",
    )
    normalized = web_db._extract_company_runs_from_payload(
        "942b82c0",
        {
            "mode": "batch",
            "summary_rows": [
                {
                    "company_name": "a22e3360c47b-varieneai-pitch-deck",
                    "startup_slug": "a22e3360c47b-varieneai-pitch-deck",
                    "decision": "watch",
                }
            ],
        },
        created_at="2026-05-15T10:47:00+00:00",
        mode="pitchdeck",
    )

    assert failed == []
    assert normalized[0]["company_name"] == "VarieneAI"
    assert normalized[0]["company_key"] == "name:varieneai"


def test_failure_result_does_not_create_company_records(monkeypatch, guardrail_modules) -> None:
    web_db = guardrail_modules.db
    called_tables: list[str] = []

    class FakeClient:
        def table(self, name):
            called_tables.append(name)
            raise RuntimeError("no table writes expected in this test")

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "upsert_job", lambda *args, **kwargs: "job-uuid")

    result = web_db.persist_company_failure_result(
        job_id_legacy="5199998a",
        result_row={
            "company": SimpleNamespace(name="variene.ai", domain=None),
            "slug": "variene-ai",
            "analysis_status": "error",
            "final_state": {},
            "evidence_store": SimpleNamespace(
                startup_slug="variene-ai",
                chunks=[SimpleNamespace(chunk_id="1", text="x")],
            ),
        },
        company_payload={"company_name": "variene.ai", "decision": "error"},
        run_config={"input_mode": "specter"},
    )

    assert result is False
    assert "companies" not in called_tables
    assert "company_runs" not in called_tables
