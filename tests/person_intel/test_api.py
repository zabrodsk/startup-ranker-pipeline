import time

from fastapi.testclient import TestClient

from web.app import app


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
