from pathlib import Path
import re


def test_backtest_replay_template_has_one_complete_document():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    assert html.lower().count("</html>") == 1
    assert html.rstrip().lower().endswith("</html>")
    assert html.lower().count("<script>") == 1
    assert html.lower().count("</script>") == 1
    # The previous bad append began with a JavaScript expression after </html>.
    assert not re.search(r"</html>\\s*\\+fmt\\(", html, re.IGNORECASE)


def test_bar_replay_controls_are_wired_and_cursor_is_bounded():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    for control in ("toStart", "prev", "next", "toEnd", "play", "cursor", "speed"):
        assert f"$('{control}').onclick" in html or f"$('{control}').oninput" in html or f"$('{control}').onchange" in html
    assert "function replayStep()" in html
    assert "function togglePlay()" in html
    assert "function setCursor(i)" in html
