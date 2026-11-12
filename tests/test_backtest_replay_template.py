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


def test_replay_chart_exposes_structure_and_trade_hit_layers():
    path = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "backtest_replay.html"
    html = path.read_text(encoding="utf-8")
    for control in ("showABC", "showSR", "showFVG", "showTrades"):
        assert f'id="{control}"' in html
    for marker in ("Bull FVG", "Bear FVG", "ENTRY", "SL", "events.forEach"):
        assert marker in html
    assert "lastResult?.trades" in html
    assert "fib_a_bar" in html and "swing_c" in html


def test_backtest_execution_events_and_costs_are_recorded():
    import pandas as pd
    from research.backtest_engine import run_backtest
    from research.strategies.base import STRATEGY_REGISTRY

    def generate(df, params):
        out = df.copy()
        out["atr"] = 1.0
        out["signal"] = [1] + [0] * (len(out) - 1)
        out["custom_stop_price"] = 98.0
        out["custom_target_price"] = 110.0
        return out

    sid = "_test_fixed_event_strategy"
    STRATEGY_REGISTRY[sid] = {
        "id": sid, "name": "Test", "description": "Test only",
        "default_params": {}, "param_schema": {}, "generate_signals": generate,
    }
    try:
        rows = [(100,101,99,100),(100,102,99,101),(100,101,99,100),(97,99,96,98),(98,99,97,98),(98,99,97,98)]
        bars = pd.DataFrame(rows, columns=["Open","High","Low","Close"],
                            index=pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC"))
        bars["Volume"] = 100
        result = run_backtest(bars, sid, initial_equity=10000, risk_pct=1,
                              spread_points=0.4, commission_pct=0, slippage_points=0.1)
        trade = result["trades"][0]
        assert trade["reason"] == "SL_GAP"
        assert [event["type"] for event in trade["events"]] == ["ENTRY", "SL"]
        assert trade["exit_price"] < 97
    finally:
        STRATEGY_REGISTRY.pop(sid, None)


def test_backtest_closes_open_position_at_end_of_data():
    import pandas as pd
    from research.backtest_engine import run_backtest
    from research.strategies.base import STRATEGY_REGISTRY

    def generate(df, params):
        out = df.copy()
        out["atr"] = 1.0
        out["signal"] = [1] + [0] * (len(out) - 1)
        out["custom_stop_price"] = 90.0
        out["custom_target_price"] = 110.0
        return out

    sid = "_test_open_position_eod"
    STRATEGY_REGISTRY[sid] = {
        "id": sid, "name": "Test", "description": "Test only",
        "default_params": {}, "param_schema": {}, "generate_signals": generate,
    }
    try:
        rows = [(100,101,99,100),(100,102,99,101),(101,103,100,102),(102,104,101,103),(103,105,102,104),(104,106,103,105)]
        bars = pd.DataFrame(rows, columns=["Open","High","Low","Close"],
                            index=pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC"))
        bars["Volume"] = 100
        result = run_backtest(bars, sid, initial_equity=10000, risk_pct=1,
                              spread_points=0, commission_pct=0, slippage_points=0)
        assert result["metrics"]["num_trades"] == 1
        assert result["trades"][0]["reason"] == "EOD"
        assert [event["type"] for event in result["trades"][0]["events"]] == ["ENTRY", "EOD"]
        assert result["metrics"]["final_equity"] > 10000
    finally:
        STRATEGY_REGISTRY.pop(sid, None)


def test_replay_engine_checks_stop_or_target_on_entry_bar():
    import pandas as pd
    from research.backtest_engine import run_backtest
    from research.strategies.base import STRATEGY_REGISTRY

    def generate(df, params):
        out = df.copy()
        out["atr"] = 1.0
        out["signal"] = [1] + [0] * (len(out) - 1)
        out["custom_stop_price"] = 98.0
        out["custom_target_price"] = 110.0
        return out

    sid = "_test_entry_bar_stop"
    STRATEGY_REGISTRY[sid] = {"id":sid,"name":"Test","description":"Test",
        "default_params":{},"param_schema":{},"generate_signals":generate}
    try:
        rows=[(100,101,99,100),(100,101,97,98),(98,99,97,98),(98,99,97,98),(98,99,97,98),(98,99,97,98)]
        bars=pd.DataFrame(rows,columns=["Open","High","Low","Close"],
            index=pd.date_range("2026-01-01",periods=len(rows),freq="5min",tz="UTC"))
        bars["Volume"]=100
        result=run_backtest(bars,sid,initial_equity=10000,risk_pct=1,
            spread_points=0,commission_pct=0,slippage_points=0)
        assert result["trades"][0]["reason"]=="SL"
        assert result["trades"][0]["entry_time"]==result["trades"][0]["exit_time"]
        assert [e["type"] for e in result["trades"][0]["events"]]==["ENTRY","SL"]
    finally:
        STRATEGY_REGISTRY.pop(sid,None)
