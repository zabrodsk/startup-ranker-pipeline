from pathlib import Path


def test_luna_is_the_frontend_pipeline_fallback_with_matching_auto_effort() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "if (modelId === 'gpt-5.6-luna')" in html
    assert "decomposition: 'medium'" in html
    assert "answering: 'low'" in html
    assert "generation: 'none'" in html
    assert "evaluation: 'medium'" in html
    assert "ranking: 'high'" in html
    assert html.count("findKey('openai', 'gpt-5.6-luna')") == 5


def test_muse_contributor_is_the_frontend_default_with_matching_auto_sampling() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "if (modelId === 'muse-spark-1.2-contributor')" in html
    assert "decomposition: 'low'" in html
    assert "answering: 'low'" in html
    assert "generation: 'minimal'" in html
    assert "evaluation: 'medium'" in html
    assert "ranking: 'high'" in html
    assert "decomposition: 0.2" in html
    assert "answering: 0.2" in html
    assert "generation: 0.7" in html
    assert "evaluation: 0.1" in html
    assert "ranking: 0.1" in html
    assert html.count("findKey('meta', 'muse-spark-1.2-contributor')") == 5
