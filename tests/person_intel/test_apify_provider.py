from agent.person_intel.providers.apify_mcp import ApifyMcpPersonProvider


def test_apify_provider_blocks_non_harvest_actor(monkeypatch) -> None:
    monkeypatch.setenv("APIFY_TOKEN", "token")
    provider = ApifyMcpPersonProvider()

    items, err = provider._run_actor_items(  # noqa: SLF001
        "dev_fusion/Linkedin-Profile-Scraper",
        {"profileUrls": ["https://www.linkedin.com/in/example"]},
    )

    assert items == []
    assert err is not None
    assert "Blocked non-harvest actor" in err


def test_apify_provider_reports_actor_metadata() -> None:
    provider = ApifyMcpPersonProvider()
    assert provider.get_actor_used() == "harvestapi/linkedin-profile-scraper"
    assert provider.get_actor_blocked() is False
