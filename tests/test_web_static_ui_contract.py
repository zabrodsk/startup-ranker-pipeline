from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "static" / "index.html"
MATCHBOOK_CSS = Path(__file__).resolve().parents[1] / "web" / "static" / "matchbook.css"


def _html() -> str:
    return INDEX_HTML.read_text()


def test_analyst_screen_and_control_contract_is_preserved() -> None:
    html = _html()

    for screen_id in (
        "password-screen",
        "upload-screen",
        "info-screen",
        "analysis-screen",
        "companies-screen",
        "leadgen-screen",
        "processing-screen",
        "results-screen",
    ):
        assert f'id="{screen_id}"' in html

    for control_id in (
        "login-form",
        "login-request-code",
        "login-unlock-button",
        "file-input",
        "specter-urls-input",
        "run-name-input",
        "instructions-input",
        "analyze-btn",
        "stop-btn",
        "companies-search-input",
        "companies-sort-select",
        "leadgen-status-filter",
        "settings-modal-overlay",
        "feedback-bubble",
        "feedback-modal-overlay",
        "rankings-sort-select",
    ):
        assert f'id="{control_id}"' in html


def test_frontend_api_and_persistence_contract_is_preserved() -> None:
    html = _html()

    for api_fragment in (
        "/api/public-auth-config",
        "/api/check-session",
        "/api/config",
        "/api/login",
        "/api/upload",
        "/api/upload-urls",
        "/api/analyze/${currentJobId}",
        "/api/status/${currentJobId}",
        "/api/jobs",
        "/api/jobs/${jobId}/control",
        "/api/jobs/${jobId}/log",
        "/api/analyses/${jobId}",
        "/api/company-runs/summary",
        "/api/company-runs/detail/",
        "/api/leadgen/intakes",
        "/api/settings/vc-strategy",
        "/api/feedback",
    ):
        assert api_fragment in html

    for storage_key in (
        "session_id",
        "active_job_id",
        "recent_runs_v2",
        "vc_investment_strategy",
        "companies_sidebar_collapsed_v1",
        "company_assessment_collapsed_v1",
        "onboarding_welcome_dismissed",
        "input_mode_callout_dismissed",
    ):
        assert storage_key in html


def test_navigation_and_dynamic_workflow_hooks_are_preserved() -> None:
    html = _html()

    for hook in (
        'data-nav-section="analysis"',
        'data-nav-section="companies"',
        'data-nav-section="leadgen"',
        'data-nav-section="new-analysis"',
        "data-job-open",
        "data-job-log",
        "data-job-stop",
        "data-leadgen-approve-selected",
        "data-leadgen-approve-all",
        "data-leadgen-reject-selected",
        "data-person-submit",
        "data-person-refresh",
        "data-feedback-ui",
    ):
        assert hook in html

    assert "function claimStartupNavigation()" in html
    assert "function syncPrimaryHeaderNav(" in html
    assert "function renderResults(" in html
    assert "function renderSavedCompaniesWindow(" in html
    assert "function renderLeadgenPage(" in html
    assert "function submitFeedback(" in html


def test_matchbook_theme_is_an_isolated_presentation_layer() -> None:
    html = _html()
    css = MATCHBOOK_CSS.read_text()

    assert '<body class="matchbook">' in html
    assert '<link href="/static/matchbook.css" rel="stylesheet">' in html
    assert "DM+Serif+Display" in html
    assert "JetBrains+Mono" in html
    assert "body.matchbook" in css
    assert "--bg-deep: #faf9f2" in css
    assert "--green: #2e4038" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "url(" not in css
    assert '<button type="button" id="settings-open"' in html
