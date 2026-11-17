from research.portfolio_classification import enrich_portfolio


def test_sector_and_theme_exposures_use_market_value_and_do_not_look_through_etfs():
    positions = [
        {"symbol": "NVDA", "market_value": 600.0},
        {"symbol": "SMH", "market_value": 300.0},
        {"symbol": "SOLUSD", "market_value": 100.0},
        {"symbol": "UNKNOWN", "market_value": 50.0},
    ]
    result = enrich_portfolio(positions, account_equity=1200.0)

    assert result["classification_coverage_pct"] == 75.0
    assert result["classification_unmapped"] == ["UNKNOWN"]
    sectors = {row["sector"]: row for row in result["sector_exposure"]}
    assert sectors["Information Technology"]["market_value"] == 600.0
    assert sectors["ETF / Fund"]["market_value"] == 300.0
    assert sectors["Crypto"]["market_value"] == 100.0
    assert sectors["Unclassified"]["market_value"] == 50.0
    assert sectors["Information Technology"]["equity_weight_pct"] == 50.0
    assert sectors["ETF / Fund"]["gross_weight_pct"] == 28.57
    assert result["positions"][1]["instrument_type"] == "ETF"
    assert result["positions"][1]["theme"] == "Semiconductors and equipment"
    assert "holdings data is supplied" in result["methodology"]


def test_instrument_and_theme_allocations_are_returned():
    result = enrich_portfolio([
        {"symbol": "IONQ", "market_value": 75.0},
        {"symbol": "WQTM", "market_value": 25.0},
    ], account_equity=100.0)

    assert result["theme_exposure_detailed"][0]["theme"] == "Quantum computing"
    assert {row["instrument_type"] for row in result["instrument_exposure"]} == {"Equity", "ETF"}
