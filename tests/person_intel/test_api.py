import time

import pytest
from fastapi.testclient import TestClient

import web.app as _web_app
from web.app import app


@pytest.fixture(autouse=True)
def _session_secret_configured(monkeypatch):
    """Pin a session secret so /api/login mints a valid session (the app fails
    closed when SESSION_SECRET is unset)."""
    monkeypatch.setattr(_web_app, "SESSION_SECRET", "test-session-secret")


def test_person_profile_job_status_model_instantiates() -> None:
    from web.app import PersonProfileJobStatus

    status = PersonProfileJobStatus(job_id="job-123", status="pending")

    assert status.job_id == "job-123"
    assert status.status == "pending"


def test_person_profile_job_lifecycle(monkeypatch) -> None:
    from web import app as web_app_module
    from agent.pipeline.state.schemas import (
        PersonClaim,
        PersonClaimEvidence,
        PersonProfileOutput,
        PersonProfileSections,
        PersonSubject,
    )

    async def fake_build_profile(self, req):
        profile = PersonProfileOutput(
            subject=PersonSubject(
                primary_profile_url=req.primary_profile_url,
                normalized_profile_url=req.primary_profile_url,
            ),
            sections=PersonProfileSections(
                interests_lifestyle="One. Two.",
                strengths=[f"✅ Strength {i}" for i in range(1, 6)],
                more_details="Details.",
                biggest_achievements=["A", "B", "C"],
                values_beliefs="One. Two.",
                key_points=["K1", "K2", "K3", "K4", "K5"],
                coolest_fact="Fact.",
                top_risk="Main uncertainty is limited evidence.",
            ),
            claims=[
                PersonClaim(
                    claim_id="claim_1",
                    text="Fact",
                    section="key_points",
                    evidence=[
                        PersonClaimEvidence(
                            url=req.primary_profile_url,
                            snippet_or_field="field: value",
                            source_type="apify_profile",
                            retrieved_at="2026-03-04T00:00:00Z",
                        )
                    ],
                    confidence=0.8,
                    timestamp="2026-03-04T00:00:00Z",
                    status="supported",
                )
            ],
            unknowns=[],
            provenance_index=[],
        )
        return profile, "## INTERESTS & LIFESTYLE\nOne. Two.\n", "abc"

    monkeypatch.setattr(
        web_app_module,
        "_person_service",
        type("S", (), {"build_profile": fake_build_profile})(),
    )

    with TestClient(app) as client:
        login = client.post('/api/login', json={'password': '9876'})
        assert login.status_code == 200
        session_id = login.json()['session_id']
        client.cookies.set('session_id', session_id)

        created = client.post(
            '/api/person-profile/jobs',
            json={'primary_profile_url': 'https://www.linkedin.com/in/example'},
        )
        assert created.status_code == 200
        job_id = created.json()['job_id']

        for _ in range(20):
            status = client.get(f'/api/person-profile/status/{job_id}')
            assert status.status_code == 200
            payload = status.json()
            if payload['status'] == 'done':
                assert payload['result']['profile_json']['claims']
                return
            time.sleep(0.05)

        raise AssertionError('Person job did not complete in time')


def test_person_profile_job_resumes_from_persisted_state(monkeypatch) -> None:
    from web import app as web_app_module
    from agent.pipeline.state.schemas import (
        PersonClaim,
        PersonClaimEvidence,
        PersonProfileOutput,
        PersonProfileSections,
        PersonSubject,
    )

    async def fake_build_profile(self, req):
        profile = PersonProfileOutput(
            subject=PersonSubject(
                primary_profile_url=req.primary_profile_url,
                normalized_profile_url=req.primary_profile_url,
            ),
            sections=PersonProfileSections(
                interests_lifestyle="One. Two.",
                strengths=[f"✅ Strength {i}" for i in range(1, 6)],
                more_details="Details.",
                biggest_achievements=["A", "B", "C"],
                values_beliefs="One. Two.",
                key_points=["K1", "K2", "K3", "K4", "K5"],
                coolest_fact="Fact.",
                top_risk="Main uncertainty is limited evidence.",
            ),
            claims=[
                PersonClaim(
                    claim_id="claim_1",
                    text="Fact",
                    section="key_points",
                    evidence=[
                        PersonClaimEvidence(
                            url=req.primary_profile_url,
                            snippet_or_field="field: value",
                            source_type="apify_profile",
                            retrieved_at="2026-03-04T00:00:00Z",
                        )
                    ],
                    confidence=0.8,
                    timestamp="2026-03-04T00:00:00Z",
                    status="supported",
                )
            ],
            unknowns=[],
            provenance_index=[],
        )
        return profile, "## INTERESTS & LIFESTYLE\nOne. Two.\n", "abc"

    class FakeDb:
        def __init__(self) -> None:
            self.upserts: list[dict[str, object]] = []

        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_person_profile_job(job_id):
            if job_id != "resume-123":
                return None
            return {
                "person_job_id": job_id,
                "status": "running",
                "progress": "Collecting evidence...",
                "request_payload": {
                    "primary_profile_url": "https://www.linkedin.com/in/example",
                    "company_slug": "apaleo",
                    "person_key": "apaleo:founder:3",
                },
                "result_payload": None,
                "error": None,
            }

        def upsert_person_profile_job(self, job_id, **kwargs):
            self.upserts.append({"job_id": job_id, **kwargs})

        def prune_person_profile_jobs(self, company_slug, person_key, keep_person_job_id=None):
            self.pruned = (company_slug, person_key, keep_person_job_id)

    fake_db = FakeDb()
    monkeypatch.setattr(
        web_app_module,
        "_person_service",
        type("S", (), {"build_profile": fake_build_profile})(),
    )
    monkeypatch.setattr(web_app_module, "db", fake_db)
    web_app_module._person_jobs.clear()
    web_app_module._person_job_tasks.clear()

    with TestClient(app) as client:
        login = client.post('/api/login', json={'password': '9876'})
        assert login.status_code == 200
        session_id = login.json()['session_id']
        client.cookies.set('session_id', session_id)

        for _ in range(20):
            status = client.get('/api/person-profile/status/resume-123')
            assert status.status_code == 200
            payload = status.json()
            if payload['status'] == 'done':
                assert payload['result']['profile_json']['claims']
                assert any(item["progress"] == "Resuming after restart..." for item in fake_db.upserts)
                return
            assert payload['status'] != 'error', f"Resumed job failed: {payload.get('error')}"
            time.sleep(0.05)

        raise AssertionError('Persisted person job did not resume in time')
