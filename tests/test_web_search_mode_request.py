"""PR2: AnalyzeRequest.web_search_mode + the per-run mode derivation.

The request boundary resolves a concrete mode once: an explicit web_search_mode
is authoritative; otherwise the legacy use_web_search bool maps True->"full",
False->"off". The worker plumbing then keeps use_web_search == (mode != "off").
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from web.app import AnalyzeRequest, _request_web_search_mode


def test_explicit_mode_wins_over_use_web_search():
    assert _request_web_search_mode("targeted", False) == "targeted"
    assert _request_web_search_mode("targeted", True) == "targeted"
    assert _request_web_search_mode("off", True) == "off"
    assert _request_web_search_mode("full", False) == "full"


def test_legacy_bool_maps_when_mode_absent():
    assert _request_web_search_mode(None, True) == "full"
    assert _request_web_search_mode(None, False) == "off"


def test_use_web_search_effective_is_consistent_with_mode():
    # off is the only mode that disables the legacy search plumbing.
    for mode, expect_on in (("off", False), ("targeted", True), ("full", True)):
        resolved = _request_web_search_mode(mode, False)
        assert (resolved != "off") is expect_on


def test_omitting_both_defaults_off():
    req = AnalyzeRequest()
    assert req.web_search_mode is None
    assert req.use_web_search is False
    assert _request_web_search_mode(req.web_search_mode, req.use_web_search) == "off"


def test_analyze_request_accepts_valid_modes():
    for mode in ("off", "targeted", "full"):
        assert AnalyzeRequest(web_search_mode=mode).web_search_mode == mode


def test_analyze_request_rejects_invalid_mode():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalyzeRequest(web_search_mode="banana")
