from research.etf_lookthrough import build_etf_lookthrough


def test_etf_lookthrough_uses_dated_top_holdings_and_preserves_uncovered_weight():
    result = build_etf_lookthrough([
        {"symbol": "SMH", "market_value": 1000},
        {"symbol": "NVDA", "market_value": 200},
        {"symbol": "WQTM", "market_value": 500},
    ])
    funds = {row["fund"]: row for row in result["funds"]}
    assert funds["SMH"]["as_of"] == "2026-11-16"
    assert funds["SMH"]["holdings"][0]["symbol"] == "NVDA"
    assert funds["SMH"]["unattributed_value_usd"] > 0
    nvda = next(row for row in result["underlying_overlap"] if row["symbol"] == "NVDA")
    assert nvda["direct_market_value_usd"] == 200
    assert nvda["lookthrough_etf_exposure_usd"] == 174.4
    assert nvda["combined_economic_exposure_usd"] == 374.4
    assert nvda["held_via_funds"] == ["SMH"]


def test_unsupported_funds_are_not_guessed_or_looked_through():
    result = build_etf_lookthrough([
        {"symbol": "QQQI", "market_value": 300},
        {"symbol": "SOXX", "market_value": 400},
        {"symbol": "IBIT", "market_value": 500},
    ])
    assert result["funds"] == []
    assert result["underlying_overlap"] == []
    assert result["unsupported_held_funds"] == ["IBIT", "QQQI", "SOXX"]
