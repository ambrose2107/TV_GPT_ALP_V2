from datetime import datetime, timedelta, timezone

from research import portfolio_research as research


def test_clean_filing_text_removes_scripts_and_collapses_whitespace():
    html = "<html><body><script>not evidence</script><h1>Revenue</h1><p> rose   12% </p></body></html>"
    text = research._clean_filing_text(html)
    assert "not evidence" not in text
    assert "Revenue rose 12%" in text


def test_sector_rotation_snapshot_uses_spy_relative_returns(monkeypatch):
    monkeypatch.setattr(research, "_alpaca_credentials", lambda: ("key", "secret"))

    def rows(start, step):
        return [{"c": start + step * i} for i in range(70)]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bars": {"XLK": rows(100.0, 1.0), "SPY": rows(100.0, 0.5)}}

    monkeypatch.setattr(research.requests, "get", lambda *args, **kwargs: FakeResponse())
    result = research.sector_rotation_snapshot()

    assert result["available"] is True
    assert len(result["sectors"]) == 1
    tech = result["sectors"][0]
    assert tech["sector"] == "Technology"
    assert tech["relative_5d_pct"] > 0
    assert tech["relative_1m_pct"] > 0
    assert tech["relative_3m_pct"] > 0
