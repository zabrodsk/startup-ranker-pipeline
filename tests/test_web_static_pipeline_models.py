from pathlib import Path


def test_luna_is_the_frontend_pipeline_fallback_with_matching_auto_effort() -> None:
    html = (Path(__file__).resolve().parents[1] / "web" / "static" / "index.html").read_text()

    assert "if (modelId === 'gpt-5.6-luna')" in html
    assert "decomposition: 'low'" in html
    assert "answering: 'low'" in html
    assert "generation: 'none'" in html
    assert "evaluation: 'medium'" in html
    assert "ranking: 'high'" in html
    assert html.count("findKey('openai', 'gpt-5.6-luna')") == 5
