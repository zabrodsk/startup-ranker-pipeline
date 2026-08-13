import re
from pathlib import Path


def test_companies_sort_uses_latest_instead_of_alphabetical() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()
    match = re.search(
        r'<select id="companies-sort-select" class="companies-sidebar-sort-select">(.*?)</select>',
        html,
        re.S,
    )

    assert match is not None
    companies_sort_html = match.group(1)
    assert '<option value="latest">LATEST</option>' in companies_sort_html
    assert '<option value="alphabetical">ALPHABETICAL</option>' not in companies_sort_html


def test_specter_url_mode_shows_deep_team_profiles_toggle() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert 'id="fetch-full-team-toggle"' in html
    assert "Fetch deep team profiles (Specter)" in html
    assert "function syncUploadOptionToggles()" in html
    assert "const haveSpecterUrls = inputMode === 'specter' && specterUrls.length > 0;" in html
    assert "(inputMode === 'pitchdeck' || haveSpecterUrls) ? 'block' : 'none'" in html
    assert "syncUploadOptionToggles();\n    updateAnalyzeButtonState();" in html
    assert "fetch_full_team: fetchFullTeam" in html


def test_scoring_signals_are_summarized_not_dumped() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "function buildStructuredSignalsSummary(signals)" in html
    assert "<ul>${rows.join('')}</ul>" in html
    assert "Client / traction:" in html
    assert "Founder archetype evidence:" in html
    assert "Specter team highlights:" in html
    assert "duplicate Specter highlights hidden" not in html
    assert "signals.specter_highlights.map(h => h.label || h).join(', ')" not in html


def test_score_and_bucket_are_authoritative_and_binary_decision_is_advisory() -> None:
    root = Path(__file__).resolve().parents[1]
    index_html = (root / "web" / "static" / "index.html").read_text()
    portal_html = (root / "web" / "static" / "portal.html").read_text()

    assert "Authoritative assessment" in index_html
    assert "Advisory recommendation" in index_html
    assert "Score and bucket are authoritative" in index_html
    assert "decision labels" not in index_html

    assert "Authoritative assessment" in portal_html
    assert "Advisory recommendation" in portal_html
    assert "Scoring & assessment" in portal_html
    assert "Scoring & decision" not in portal_html
    assert "<strong>Decision:</strong>" not in portal_html


def test_company_rank_cards_render_website_and_specter_links() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "const SPECTER_PROFILE_BASE_URL = 'https://app.tryspecter.com/signals/company/feed/';" in html
    assert "function buildCompanyLinkControlsHtml(...sources)" in html
    assert "function buildRankCompanyTitleHtml(companyName, ...sources)" in html
    assert "target=\"_blank\" rel=\"noopener noreferrer\"" in html
    assert "buildSpecterCompanyProfileUrl(specterCompanyId)" in html
    assert "buildRankCompanyTitleHtml(companyName, result, summary, company, run)" in html
    assert "buildRankCompanyTitleHtml(row.company_name || row.startup_slug, row)" in html


def test_leadgen_intake_screen_uses_human_approval_api() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert 'id="leadgen-screen"' in html
    assert "function openLeadgenPage()" in html
    assert "`/api/leadgen/intakes?status=${encodeURIComponent(leadgenStatusFilter)}&limit=100`" in html
    assert "`/api/leadgen/intakes/${encodeURIComponent(intakeId)}/approve`" in html
    assert "Approve Selected & Start" in html
    assert "Approve All Eligible & Start" in html
    assert "let leadgenSortMode = 'source';" in html
    assert "function leadgenSourceLabel(lead)" in html
    assert '<th>Source</th>' in html
    assert '<option value="source" ${leadgenSortMode === \'source\' ? \'selected\' : \'\'}>Source</option>' in html
    assert '<th>Score</th><th>Bucket</th>' not in html
    assert "function estimateSpecterMcpMinimumCalls(companyCount)" in html
    assert "Specter quota remaining: unknown. Selected minimum:" in html
    assert "Approve-all minimum:" in html


def test_specter_url_intake_shows_quota_unknown_warning() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "specter-quota-warning" in html
    assert "This run requires at least ${minimumCalls} MCP calls, including one preflight call." in html
    assert "error.status = res.status;" in html
    assert "if (err?.status) throw err;" in html
    assert "if (err?.status) {" in html


def test_startup_navigation_does_not_override_companies_click() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "let startupNavigationInProgress = true;" in html
    assert "let startupNavigationClaimedByUser = false;" in html
    assert "function claimStartupNavigation()" in html

    bootstrap_guard = re.search(
        r"if \(startupNavigationClaimedByUser\) \{\s*"
        r"startupNavigationInProgress = false;\s*"
        r"syncFeedbackButtonVisibility\(\);\s*"
        r"return;\s*"
        r"\}\s*"
        r"const resumed = await resumeActiveJob\(\);",
        html,
        re.S,
    )
    assert bootstrap_guard is not None

    companies_handler = re.search(
        r"el\.addEventListener\('click', \(\) => \{\s*"
        r"claimStartupNavigation\(\);\s*"
        r"openCompaniesPage\(\);",
        html,
        re.S,
    )
    assert companies_handler is not None
