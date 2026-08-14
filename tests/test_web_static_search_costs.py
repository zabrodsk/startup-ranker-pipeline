from pathlib import Path


PORTAL_HTML = Path(__file__).parents[1] / "web" / "static" / "portal.html"


def test_admin_portal_renders_combined_web_search_costs() -> None:
    html = PORTAL_HTML.read_text(encoding="utf-8")

    assert "const serper = cs.serper_usd;" in html
    assert "Serper: ${fmtUsd(serper)}" in html
    assert "const serper = rc && rc.serper_search ? rc.serper_search : {};" in html
    assert '>Web:</span>' in html
    assert "(rc.perplexity_usd || 0) + (rc.serper_usd || 0)" in html
    assert "Serper {fmtInt(serper.requests || 0)}" in html
