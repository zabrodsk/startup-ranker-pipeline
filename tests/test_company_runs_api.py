import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from web import app as web_app


def test_company_run_summary_endpoint_returns_lightweight_rows(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)

    def fake_summary(*, limit: int, offset: int, sort: str):
        assert (limit, offset, sort) == (200, 0, "latest")
        return {
            "companies": [
                {
                    "company_lookup_key": "name:apify",
                    "company_name": "Apify",
                    "run_count": 3,
                    "latest_job_id": "job-apify",
                    "latest_score": 82.0,
                    "latest_run_at": "2026-03-10T10:00:00Z",
                },
                {
                    "company_lookup_key": "name:apaleo",
                    "company_name": "Apaleo",
                    "run_count": 2,
                    "latest_job_id": "job-apaleo",
                    "latest_score": 79.0,
                    "latest_run_at": "2026-03-09T10:00:00Z",
                },
            ],
            "total": 2,
            "next_offset": None,
        }

    monkeypatch.setattr(web_app, "_list_company_run_summaries_for_ui", fake_summary)

    with TestClient(web_app.app) as client:
        response = client.get("/api/company-runs/summary?limit=200&offset=0&sort=latest")

    assert response.status_code == 200
    payload = response.json()
    assert [row["company_name"] for row in payload["companies"]] == ["Apify", "Apaleo"]
    assert all("result_payload" not in row and "runs" not in row for row in payload["companies"])


def test_company_run_detail_endpoint_returns_selected_company_only(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    requested_keys: list[str] = []

    def fake_detail(company_lookup_key: str):
        requested_keys.append(company_lookup_key)
        return {
            "company_lookup_key": company_lookup_key,
            "company_name": "Apify",
            "runs": [
                {
                    "job_id": "job-apify",
                    "created_at": "2026-03-10T10:00:00Z",
                    "results": {
                        "company_name": "Apify",
                        "company_url": "https://apify.com",
                        "specter_company_id": "specter-apify",
                        "specter_profile_url": "https://app.tryspecter.com/signals/company/feed/specter-apify",
                    },
                }
            ],
        }

    monkeypatch.setattr(web_app, "_load_company_run_detail_for_ui", fake_detail)

    with TestClient(web_app.app) as client:
        response = client.get("/api/company-runs/detail/name%3Aapify")

    assert response.status_code == 200
    assert requested_keys == ["name:apify"]
    assert response.json()["company"]["runs"][0]["job_id"] == "job-apify"
    assert response.json()["company"]["runs"][0]["results"]["company_url"] == "https://apify.com"
    assert response.json()["company"]["runs"][0]["results"]["specter_profile_url"] == (
        "https://app.tryspecter.com/signals/company/feed/specter-apify"
    )
