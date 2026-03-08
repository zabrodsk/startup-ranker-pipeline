from pathlib import Path

import pandas as pd

from agent.ingest.specter_ingest import ingest_specter
from web.app import _detect_specter_csvs


def test_ingest_specter_companies_only_returns_all_rows(tmp_path: Path) -> None:
    companies_path = tmp_path / "specter-export.csv"
    pd.DataFrame(
        [
            {"Company Name": "Alpha", "Industry": "SaaS", "Description": "A"},
            {"Company Name": "Beta", "Industry": "Fintech", "Description": "B"},
            {"Company Name": "Gamma", "Industry": "Health", "Description": "C"},
        ]
    ).to_csv(companies_path, index=False)

    results = ingest_specter(companies_path)

    assert [company.name for company, _ in results] == ["Alpha", "Beta", "Gamma"]


def test_detect_specter_csvs_by_headers_without_people_file(tmp_path: Path) -> None:
    companies_path = tmp_path / "generic-upload.csv"
    pd.DataFrame(
        [
            {
                "Company Name": "Alpha",
                "Founders": "[]",
                "Industry": "SaaS",
                "Domain": "alpha.com",
            }
        ]
    ).to_csv(companies_path, index=False)

    detected = _detect_specter_csvs(tmp_path, [companies_path.name])

    assert detected == {"companies": str(companies_path)}
