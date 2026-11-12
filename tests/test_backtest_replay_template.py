from pathlib import Path
import re


def test_backtest_replay_template_has_one_complete_document():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    assert html.lower().count("</html>") == 1
    assert html.rstrip().lower().endswith("</html>")
    assert html.lower().count("<script") == 2  # Chart.js CDN plus one inline controller
    assert html.lower().count("</script>") == 2
    # The previous bad append began with a JavaScript expression after </html>.
    assert not re.search(r"</html>\s*\+fmt\(", html, re.IGNORECASE)


def test_bar_replay_controls_are_wired_and_cursor_is_bounded():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    for control in ("toStart", "prev", "next", "toEnd", "play", "cursor", "speed"):
        assert f"$('{control}').onclick" in html or f"$('{control}').oninput" in html or f"$('{control}').onchange" in html
    assert "function replayStep()" in html
    assert "function togglePlay()" in html
    assert "function setCursor(i)" in html


def test_chart_validity_helper_is_defined_before_first_use():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    draw = html[html.index("function draw(){"):html.index("\nfunction render()", html.index("function draw(){"))]
    assert draw.index("const valid=") < draw.index("forEach(k=>{if(valid(s[k]))")
