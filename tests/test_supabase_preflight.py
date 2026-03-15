import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, client, table_name: str):
        self.client = client
        self.table_name = table_name
        self._filters: dict[str, object] = {}
        self._delete = False

    def select(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def upsert(self, payload, **_kwargs):
        self.client.disposable_job_id = payload["job_id_legacy"]
        self.client.calls.append(("upsert", self.table_name, payload["job_id_legacy"]))
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def delete(self):
        self._delete = True
        return self

    def execute(self):
        self.client.calls.append(("execute", self.table_name, dict(self._filters), self._delete))
        if self.table_name in self.client.fail_tables:
            raise RuntimeError(f"permission denied for table {self.table_name}")
        if self._delete:
            return FakeResponse([])
        if (
            self.table_name == "jobs"
            and self._filters.get("job_id_legacy") == self.client.disposable_job_id
        ):
            return FakeResponse([{"job_id_legacy": self.client.disposable_job_id}])
        return FakeResponse([{}])


class FakeClient:
    def __init__(self, fail_tables: set[str] | None = None):
        self.calls: list[tuple] = []
        self.disposable_job_id: str | None = None
        self.fail_tables = fail_tables or set()

    def table(self, table_name: str):
        return FakeQuery(self, table_name)


def test_run_preflight_checks_reads_expected_tables_and_cleans_up() -> None:
    import web.supabase_preflight as preflight

    client = FakeClient()
    messages: list[str] = []

    job_id = preflight.run_preflight_checks(client, print_fn=messages.append)

    assert job_id.startswith("rls-preflight-")
    assert client.disposable_job_id == job_id
    assert messages[:5] == [
        "[check] read jobs",
        "[check] read analyses",
        "[check] read company_runs",
        "[check] read job_controls",
        "[check] read person_profile_jobs",
    ]
    assert any(call[0] == "upsert" and call[1] == "jobs" for call in client.calls)
    assert any(call[1] == "jobs" and call[2].get("job_id_legacy") == job_id for call in client.calls if call[0] == "execute")
    assert any(call[1] == "jobs" and call[3] is True for call in client.calls if call[0] == "execute")


def test_run_preflight_checks_fails_loudly_on_permission_errors() -> None:
    import web.supabase_preflight as preflight

    client = FakeClient(fail_tables={"analyses"})

    with pytest.raises(preflight.PreflightFailure) as exc_info:
        preflight.run_preflight_checks(client)

    assert "table=analyses" in str(exc_info.value)
    assert "permission_related=True" in str(exc_info.value)
