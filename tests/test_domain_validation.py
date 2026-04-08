"""Unit tests for Sprint 1 domain validation helpers.

These tests are purely in-process — no network, no Supabase, no FastAPI client.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pytest

from web.app import (
    _extract_email_domain,
    _normalize_domain,
    _root_domain,
    verify_email_domain,
)


# ---------------------------------------------------------------------------
# _normalize_domain
# ---------------------------------------------------------------------------

def test_normalize_domain_strips_https():
    assert _normalize_domain("https://company.com") == "company.com"


def test_normalize_domain_strips_http():
    assert _normalize_domain("http://company.com") == "company.com"


def test_normalize_domain_strips_www():
    assert _normalize_domain("www.company.com") == "company.com"


def test_normalize_domain_strips_scheme_and_www():
    assert _normalize_domain("https://www.company.com") == "company.com"


def test_normalize_domain_strips_trailing_path():
    assert _normalize_domain("https://company.com/about") == "company.com"


def test_normalize_domain_lowercases():
    assert _normalize_domain("Company.COM") == "company.com"


def test_normalize_domain_plain():
    assert _normalize_domain("company.com") == "company.com"


# ---------------------------------------------------------------------------
# _extract_email_domain
# ---------------------------------------------------------------------------

def test_extract_email_domain_basic():
    assert _extract_email_domain("user@company.com") == "company.com"


def test_extract_email_domain_uppercase():
    assert _extract_email_domain("User@Company.COM") == "company.com"


def test_extract_email_domain_subdomain():
    assert _extract_email_domain("user@mail.company.com") == "mail.company.com"


def test_extract_email_domain_missing_at():
    with pytest.raises(ValueError, match="Invalid email"):
        _extract_email_domain("notanemail")


def test_extract_email_domain_empty():
    with pytest.raises(ValueError):
        _extract_email_domain("")


def test_extract_email_domain_no_dot_in_domain():
    with pytest.raises(ValueError, match="Invalid email domain"):
        _extract_email_domain("user@localhost")


# ---------------------------------------------------------------------------
# _root_domain
# ---------------------------------------------------------------------------

def test_root_domain_two_labels():
    assert _root_domain("company.com") == "company.com"


def test_root_domain_three_labels():
    assert _root_domain("mail.company.com") == "company.com"


def test_root_domain_four_labels():
    assert _root_domain("a.mail.company.com") == "company.com"


# ---------------------------------------------------------------------------
# verify_email_domain
# ---------------------------------------------------------------------------

def test_verify_exact_match():
    verified, reason = verify_email_domain("user@company.com", "company.com")
    assert verified is True
    assert "company.com" in reason


def test_verify_with_https_company_domain():
    verified, _ = verify_email_domain("user@company.com", "https://company.com")
    assert verified is True


def test_verify_with_www_company_domain():
    verified, _ = verify_email_domain("user@company.com", "www.company.com")
    assert verified is True


def test_verify_subdomain_matches_root():
    """user@mail.company.com should match company domain company.com via root domain."""
    verified, _ = verify_email_domain("user@mail.company.com", "company.com")
    assert verified is True


def test_verify_mismatch():
    verified, reason = verify_email_domain("user@other.com", "company.com")
    assert verified is False
    assert "does not match" in reason


def test_verify_free_provider_gmail():
    verified, reason = verify_email_domain("user@gmail.com", "gmail.com")
    assert verified is False
    assert "free email provider" in reason.lower()


def test_verify_free_provider_yahoo():
    verified, reason = verify_email_domain("user@yahoo.com", "yahoo.com")
    assert verified is False
    assert "free email provider" in reason.lower()


def test_verify_free_provider_hotmail():
    verified, reason = verify_email_domain("user@hotmail.com", "somecompany.com")
    assert verified is False
    assert "free email provider" in reason.lower()


def test_verify_no_company_domain():
    verified, reason = verify_email_domain("user@company.com", None)
    assert verified is False
    assert "no domain set" in reason.lower() or "admin" in reason.lower()


def test_verify_empty_company_domain():
    verified, reason = verify_email_domain("user@company.com", "")
    assert verified is False


def test_verify_case_insensitive():
    verified, _ = verify_email_domain("User@Company.COM", "COMPANY.COM")
    assert verified is True


def test_verify_invalid_email():
    verified, reason = verify_email_domain("notanemail", "company.com")
    assert verified is False
    assert "invalid" in reason.lower()
