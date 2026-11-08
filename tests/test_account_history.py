from core.account_history import build_history_summary

def test_account_history_reconciles_cash_flows_dividends_tax_and_positions():
    activities=[
      {"id":"1","activity_type":"FILL","symbol":"AAPL","side":"buy","qty":"2","price":"100","transaction_time":"2026-01-02T10:00:00Z"},
      {"id":"2","activity_type":"FILL","symbol":"AAPL","side":"sell","qty":"1","price":"120","transaction_time":"2026-01-03T10:00:00Z"},
      {"id":"3","activity_type":"CSD","net_amount":"500","date":"2026-01-04"},
      {"id":"4","activity_type":"CSW","net_amount":"-100","date":"2026-01-05"},
      {"id":"5","activity_type":"DIV","net_amount":"3.00","symbol":"AAPL","date":"2026-01-06"},
      {"id":"6","activity_type":"DIVNRA","net_amount":"-0.45","symbol":"AAPL","date":"2026-01-06"},
      {"id":"7","activity_type":"FEE","net_amount":"-1.00","date":"2026-01-07"},
      {"id":"8","activity_type":"INT","net_amount":"0.25","date":"2026-01-08"}
    ]
    positions=[{"cost_basis":"100","market_value":"120","unrealized_pl":"20"}]
    summary,rows=build_history_summary(activities,positions,{"equity":"600","cash":"480","buying_power":"480"})
    assert summary["current_equity"]==600.0
    assert summary["current_invested_cost_basis"]==100.0
    assert summary["realized_pnl_fifo"]==20.0
    assert summary["unrealized_pnl"]==20.0
    assert summary["gross_lifetime_buys"]==200.0
    assert summary["gross_lifetime_sells"]==120.0
    assert summary["deposits"]==500.0 and summary["withdrawals"]==100.0
    assert summary["net_cash_flows"]==400.0
    assert summary["gross_dividends"]==3.0 and summary["dividend_withheld"]==0.45
    assert summary["net_dividends"]==2.55 and summary["tax_withheld"]==0.45
    assert summary["fees"]==1.0 and summary["interest"]==0.25
    assert len(rows)==len(activities)

def test_orphan_sell_does_not_create_fictitious_profit():
    summary,_=build_history_summary([{"activity_type":"FILL","symbol":"MSFT","side":"sell","qty":"3","price":"100","transaction_time":"2026-01-03T10:00:00Z"}],[],{"equity":"1000"})
    assert summary["realized_pnl_fifo"]==0.0
    assert summary["realized_fifo_unmatched_sell_qty"]==3.0


def test_analytics_template_contains_account_history_view():
    from pathlib import Path
    template = Path("dashboard/templates/dashboard.html").read_text(encoding="utf-8")
    for marker in ("avbtn-history", "av-history", "ah-equity", "ah-dividends", "ah-deposits", "ah-withdrawals", "ah-tbody", "loadAccountHistory"):
        assert marker in template, f"Analytics account-history UI is missing: {marker}"


def test_alpaca_until_timestamp_is_never_in_the_future():
    from datetime import datetime, timezone
    from core.account_history import _alpaca_until
    future = _alpaca_until("2099-12-31")
    parsed = datetime.fromisoformat(future.replace("Z", "+00:00"))
    assert parsed <= datetime.now(timezone.utc)
    past = _alpaca_until("2020-01-01")
    assert past == "2020-01-01T23:59:59Z"


def test_account_history_template_has_no_duplicate_html_or_trailing_javascript():
    from pathlib import Path
    template = Path("dashboard/templates/dashboard.html").read_text(encoding="utf-8")
    assert template.lower().count("</html>") == 1
    assert not template.split("</html>", 1)[1].strip()
    assert template.count("function accountHistoryRange(range)") == 1


def test_alpaca_activity_request_excludes_unsupported_divpwh_type():
    from pathlib import Path
    source = Path("core/account_history.py").read_text(encoding="utf-8")
    request_line = next(line for line in source.splitlines() if '"activity_types":' in line)
    assert "DIVPWH" not in request_line
    assert "DIVNRA" in request_line and "DIVFT" in request_line and "DIVTW" in request_line
