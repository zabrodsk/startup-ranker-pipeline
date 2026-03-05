"""Apify-backed provider for public profile evidence."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import certifi
import requests

from agent.person_intel.models import EvidenceRecord, PersonIntelSubject, PersonProfileJobRequest
from agent.person_intel.providers.base import PersonSourceProvider

HARVEST_PROFILE_ACTOR_ID = "harvestapi/linkedin-profile-scraper"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApifyMcpPersonProvider(PersonSourceProvider):
    """Collect evidence from Apify actors (profile + posts)."""

    def __init__(self) -> None:
        self.token = os.getenv("APIFY_TOKEN", "").strip()
        # Hard-pinned by product decision: only this actor is allowed.
        self.profile_actor_id = HARVEST_PROFILE_ACTOR_ID
        self.base_url = os.getenv("APIFY_API_BASE_URL", "https://api.apify.com/v2").rstrip("/")
        self.max_items = int(os.getenv("APIFY_MAX_ITEMS", "20"))
        self.last_errors: list[str] = []
        self.last_attempts: list[dict[str, Any]] = []

    def get_last_error_message(self) -> str | None:
        """Return latest collected Apify error, if any."""
        if not self.last_errors:
            return None
        deduped: list[str] = []
        seen: set[str] = set()
        for item in self.last_errors:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return " | ".join(deduped[:3])

    def get_last_attempts(self) -> list[dict[str, Any]]:
        return list(self.last_attempts)

    def get_actor_used(self) -> str:
        return self.profile_actor_id

    def get_actor_blocked(self) -> bool:
        if self.profile_actor_id.lower() != HARVEST_PROFILE_ACTOR_ID:
            return True
        for err in self.last_errors:
            if "Blocked non-harvest actor" in err:
                return True
        return False

    async def collect(
        self,
        request: PersonProfileJobRequest,
        subject: PersonIntelSubject,
    ) -> list[EvidenceRecord]:
        self.last_errors = []
        self.last_attempts = []
        if not self.token:
            self.last_errors.append("Missing APIFY_TOKEN.")
            return []

        records: list[EvidenceRecord] = []
        profile_items: list[dict[str, Any]] = []
        if self.profile_actor_id.lower() != HARVEST_PROFILE_ACTOR_ID:
            self.last_errors.append(
                f"Blocked non-harvest actor '{self.profile_actor_id}'. Expected '{HARVEST_PROFILE_ACTOR_ID}'."
            )
            return []
        payload = self._build_actor_payload(self.profile_actor_id, subject.normalized_profile_url)
        profile_items, actor_error = await asyncio.to_thread(
            self._run_actor_items,
            self.profile_actor_id,
            payload,
        )
        if actor_error:
            self.last_errors.append(f"{self.profile_actor_id}: {actor_error}")

        for item in profile_items:
            records.extend(self._extract_profile_records(item, subject.normalized_profile_url))

        return records

    def _build_actor_payload(self, actor_id: str, profile_url: str) -> dict[str, Any]:
        # harvestapi/linkedin-profile-scraper input schema.
        if actor_id.lower() == HARVEST_PROFILE_ACTOR_ID:
            return {
                "profileScraperMode": "Profile details no email ($4 per 1k)",
                "urls": [profile_url],
                "queries": [profile_url],
            }
        # Defensive fallback (should not be used due to hard pinning above).
        return {
            "profileUrls": [profile_url],
            "urls": [profile_url],
            "startUrls": [{"url": profile_url}],
        }

    def _run_actor_items(
        self,
        actor_id: str,
        payload: dict[str, Any],
        channel: str = "profile",
    ) -> tuple[list[dict[str, Any]], str | None]:
        if actor_id.lower() != HARVEST_PROFILE_ACTOR_ID:
            return (
                [],
                f"Blocked non-harvest actor '{actor_id}'. Only '{HARVEST_PROFILE_ACTOR_ID}' is allowed.",
            )
        actor_ref = actor_id.replace("/", "~")
        run_url = f"{self.base_url}/acts/{actor_ref}/runs"
        ssl_verify_env = os.getenv("APIFY_SSL_VERIFY", "true").strip().lower()
        verify: bool | str = False if ssl_verify_env in {"0", "false", "no"} else certifi.where()
        attempt: dict[str, Any] = {
            "channel": channel,
            "actor_id": actor_id,
            "actor_ref": actor_ref,
            "requested_at": _utc_now_iso(),
        }

        try:
            run_resp = requests.post(
                run_url,
                params={"token": self.token, "waitForFinish": 120},
                json=payload,
                timeout=180,
                verify=verify,
            )
        except requests.exceptions.SSLError as exc:
            err = f"SSL error during actor run: {exc}"
            attempt["result"] = {"success": False, "error": err}
            self.last_attempts.append(attempt)
            return ([], err)
        except requests.exceptions.RequestException as exc:
            err = f"Network error during actor run: {exc}"
            attempt["result"] = {"success": False, "error": err}
            self.last_attempts.append(attempt)
            return ([], err)

        if run_resp.status_code >= 400:
            err = f"HTTP {run_resp.status_code} during actor run: {run_resp.text[:260]}"
            attempt["result"] = {"success": False, "error": err}
            self.last_attempts.append(attempt)
            return ([], err)

        try:
            run_data = run_resp.json()
        except json.JSONDecodeError:
            err = "Invalid JSON received from actor run."
            attempt["result"] = {"success": False, "error": err}
            self.last_attempts.append(attempt)
            return ([], err)

        run_payload = run_data.get("data", {}) if isinstance(run_data, dict) else {}
        status = str(run_payload.get("status") or "").upper()
        status_msg = str(run_payload.get("statusMessage") or "").strip()
        started_at = self._parse_apify_datetime(run_payload.get("startedAt"))
        dataset_id = run_payload.get("defaultDatasetId")
        attempt["initial_run"] = {
            "run_id": run_payload.get("id"),
            "status": status,
            "status_message": status_msg,
            "started_at": run_payload.get("startedAt"),
            "finished_at": run_payload.get("finishedAt"),
            "dataset_id": dataset_id,
        }

        primary_items: list[dict[str, Any]] = []
        primary_error: str | None = None
        if dataset_id:
            primary_items, primary_error = self._fetch_dataset_items(str(dataset_id), verify)
            if primary_items:
                attempt["result"] = {
                    "success": True,
                    "source": "initial_run_dataset",
                    "items_count": len(primary_items),
                }
                self.last_attempts.append(attempt)
                return (primary_items, None)

        errors: list[str] = []
        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            errors.append(f"run ended with status {status}")
        if status_msg.startswith("❌"):
            errors.append(status_msg)
        if primary_error:
            errors.append(primary_error)
        if not dataset_id:
            errors.append("Actor run returned no dataset.")

        fallback_run = self._wait_for_followup_success_dataset(actor_ref, started_at, verify)
        if fallback_run:
            attempt["followup_run"] = fallback_run
            fallback_items, fallback_error = self._fetch_dataset_items(
                str(fallback_run.get("dataset_id")),
                verify,
            )
            if fallback_items:
                attempt["result"] = {
                    "success": True,
                    "source": "followup_run_dataset",
                    "items_count": len(fallback_items),
                }
                self.last_attempts.append(attempt)
                return (fallback_items, None)
            if fallback_error:
                errors.append(f"follow-up successful run dataset read failed: {fallback_error}")

        if errors:
            err = " | ".join(errors[:3])
            attempt["result"] = {"success": False, "error": err}
            self.last_attempts.append(attempt)
            return ([], err)

        err = "Actor run returned no usable items."
        attempt["result"] = {"success": False, "error": err}
        self.last_attempts.append(attempt)
        return ([], err)

    def _fetch_dataset_items(self, dataset_id: str, verify: bool | str) -> tuple[list[dict[str, Any]], str | None]:
        items_url = f"{self.base_url}/datasets/{dataset_id}/items"
        try:
            items_resp = requests.get(
                items_url,
                params={
                    "token": self.token,
                    "clean": "true",
                    "format": "json",
                    "limit": self.max_items,
                    "desc": "true",
                },
                timeout=120,
                verify=verify,
            )
        except requests.exceptions.SSLError as exc:
            return ([], f"SSL error while reading dataset: {exc}")
        except requests.exceptions.RequestException as exc:
            return ([], f"Network error while reading dataset: {exc}")

        if items_resp.status_code >= 400:
            return ([], f"HTTP {items_resp.status_code} while reading dataset: {items_resp.text[:260]}")

        try:
            items = items_resp.json()
        except json.JSONDecodeError:
            return ([], "Invalid JSON while reading dataset.")

        if not isinstance(items, list):
            return ([], "Dataset payload was not a JSON list.")

        for item in items:
            if isinstance(item, dict) and item.get("error"):
                return ([], str(item.get("error")))
        return ([item for item in items if isinstance(item, dict)], None)

    def _wait_for_followup_success_dataset(
        self,
        actor_ref: str,
        started_at: datetime | None,
        verify: bool | str,
    ) -> dict[str, Any] | None:
        wait_seconds = int(os.getenv("APIFY_FOLLOWUP_WAIT_SECONDS", "45"))
        if wait_seconds <= 0:
            return None
        deadline = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        runs_url = f"{self.base_url}/acts/{actor_ref}/runs"

        while datetime.now(timezone.utc) < deadline:
            try:
                runs_resp = requests.get(
                    runs_url,
                    params={"token": self.token, "limit": 15, "desc": "true"},
                    timeout=60,
                    verify=verify,
                )
            except requests.exceptions.RequestException:
                return None

            if runs_resp.status_code < 400:
                try:
                    payload = runs_resp.json()
                except json.JSONDecodeError:
                    payload = {}

                runs = payload.get("data", {}).get("items", []) if isinstance(payload, dict) else []
                if isinstance(runs, list):
                    for run in runs:
                        if not isinstance(run, dict):
                            continue
                        run_status = str(run.get("status") or "").upper()
                        if run_status != "SUCCEEDED":
                            continue
                        run_started_at = self._parse_apify_datetime(run.get("startedAt"))
                        if started_at and run_started_at and run_started_at < started_at:
                            continue
                        dataset_id = run.get("defaultDatasetId")
                        if dataset_id:
                            return {
                                "run_id": run.get("id"),
                                "status": run_status,
                                "status_message": run.get("statusMessage"),
                                "started_at": run.get("startedAt"),
                                "finished_at": run.get("finishedAt"),
                                "dataset_id": str(dataset_id),
                            }
            time.sleep(3)

        return None

    @staticmethod
    def _parse_apify_datetime(raw: Any) -> datetime | None:
        if not raw or not isinstance(raw, str):
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _extract_profile_records(self, item: dict[str, Any], fallback_url: str) -> list[EvidenceRecord]:
        url = str(item.get("linkedinUrl") or item.get("profileUrl") or item.get("url") or fallback_url)
        fields = [
            "fullName",
            "firstName",
            "lastName",
            "headline",
            "about",
            "summary",
            "currentCompany",
            "currentPosition",
            "location",
            "profilePicture",
            "profilePic",
            "profilePhoto",
            "profileImage",
            "followers",
            "connections",
            "experience",
            "education",
            "skills",
            "accomplishments",
            "certifications",
            "languages",
            "volunteering",
            "projects",
            "publications",
            "awards",
            "honors",
        ]
        records: list[EvidenceRecord] = []
        retrieved_at = _utc_now_iso()
        for field in fields:
            value = item.get(field)
            if value is None:
                continue
            snippet = self._humanize_field_value(field, value)
            if not snippet:
                continue
            records.append(
                EvidenceRecord(
                    url=url,
                    snippet_or_field=f"{field}: {snippet}",
                    source_type="apify_profile",
                    retrieved_at=retrieved_at,
                )
            )
        return records

    @staticmethod
    def _collapse_spaces(text: str) -> str:
        return " ".join((text or "").strip().split())

    def _humanize_field_value(self, field: str, value: Any) -> str:
        if isinstance(value, str):
            return self._collapse_spaces(value)[:600]
        if isinstance(value, (int, float)):
            return str(value)

        if field in {"skills", "languages"} and isinstance(value, list):
            vals = []
            for entry in value:
                if isinstance(entry, dict):
                    name = entry.get("title") or entry.get("name")
                    if name:
                        vals.append(self._collapse_spaces(str(name)))
                elif isinstance(entry, str):
                    vals.append(self._collapse_spaces(entry))
            return ", ".join(vals[:20])

        if field in {"experience", "projects", "publications"} and isinstance(value, list):
            rows: list[str] = []
            for entry in value[:8]:
                if not isinstance(entry, dict):
                    continue
                title = self._collapse_spaces(str(entry.get("title") or entry.get("name") or ""))
                company = self._collapse_spaces(str(entry.get("companyName") or entry.get("company") or ""))
                desc = self._collapse_spaces(str(entry.get("description") or ""))
                if title and company:
                    rows.append(f"{title} at {company}")
                elif title:
                    rows.append(title)
                elif company:
                    rows.append(company)
                elif desc:
                    rows.append(desc)
            return "; ".join(rows)

        if field in {"education", "certifications", "awards", "honors"} and isinstance(value, list):
            rows: list[str] = []
            for entry in value[:8]:
                if not isinstance(entry, dict):
                    continue
                a = self._collapse_spaces(str(entry.get("schoolName") or entry.get("issuer") or entry.get("name") or ""))
                b = self._collapse_spaces(str(entry.get("degreeName") or entry.get("title") or ""))
                if a and b:
                    rows.append(f"{a} ({b})")
                elif a:
                    rows.append(a)
                elif b:
                    rows.append(b)
            return "; ".join(rows)

        if isinstance(value, dict):
            prioritized = []
            for key in ("title", "name", "description", "value", "text"):
                v = value.get(key)
                if v:
                    prioritized.append(self._collapse_spaces(str(v)))
            if prioritized:
                return "; ".join(prioritized)[:600]
            return ""

        if isinstance(value, list):
            vals = [self._collapse_spaces(str(v)) for v in value if str(v).strip()]
            return ", ".join(vals[:20])[:600]

        return self._collapse_spaces(str(value))[:600]
