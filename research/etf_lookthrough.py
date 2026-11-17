"""Dated, issuer-published ETF holdings snapshots for transparent portfolio look-through.

Snapshots are deliberately limited to top holdings visible on official issuer pages.
They are not represented as complete fund holdings. Uncovered fund weight is kept
unattributed. Refresh dates and source URLs travel with every result.
"""
from collections import defaultdict

ETF_HOLDINGS = {
    "SMH": {
        "as_of": "2026-11-16",
        "source": "https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/holdings/",
        "coverage_pct": 84.28,
        "holdings": [
            ("NVDA", 17.44), ("TSM", 9.97), ("AVGO", 6.50), ("TXN", 6.31),
            ("AMD", 6.05), ("ADI", 5.15), ("ASML", 4.57), ("QCOM", 4.49),
            ("KLAC", 4.42), ("MU", 4.37), ("AMAT", 4.19), ("SKHY", 3.84),
            ("LRCX", 3.74), ("MRVL", 3.24),
        ],
    },
    "CIBR": {
        "as_of": "2026-11-16",
        "source": "https://www.ftportfolios.com/Retail/Etf/EtfHoldings.aspx?Ticker=CIBR",
        "coverage_pct": 42.21,
        "holdings": [
            ("CRWD", 8.29), ("AVGO", 7.73), ("CSCO", 7.34),
            ("PANW", 7.10), ("FTNT", 6.19), ("OKTA", 4.17),
            ("NTCT", 0.35), ("ATEN", 0.34), ("ZD", 0.24),
            ("RDWR", 0.16), ("OSPN", 0.15), ("RPD", 0.15),
        ],
    },
    "WQTM": {
        "as_of": "2026-09-25",
        "source": "https://www.wisdomtree.com/us/products/megatrends/wqtm",
        "coverage_pct": 48.09,
        "holdings": [
            ("IONQ", 6.59), ("QUANTINUUM_PRIVATE", 6.41), ("RGTI", 5.86),
            ("QBTS", 5.43), ("INFLIQTION_PRIVATE", 5.13), ("IBM", 5.01),
            ("FUJITSU", 3.69), ("GFS", 3.48), ("MSFT", 3.30), ("AMZN", 3.19),
        ],
    },
    "QQQ": {
        "as_of": "2026-11-03",
        "source": "https://www.invesco.com/qqq-etf/en/about.html",
        "coverage_pct": 46.44,
        "holdings": [
            ("AAPL", 8.13), ("NVDA", 8.02), ("MSFT", 6.20), ("AMZN", 4.36),
            ("MU", 4.28), ("TSLA", 3.49), ("AMD", 3.47), ("META", 2.97),
            ("WMT", 2.82), ("GOOGL", 2.70),
        ],
    },
}


def build_etf_lookthrough(positions):
    """Translate held ETF market value into dated underlying exposures.

    Returns a top-holdings-only estimate. The ETF wrapper is not added again to
    underlying exposure totals, and unknown/uncovered weight remains explicit.
    """
    direct = {
        str(p.get("symbol") or "").upper(): abs(float(p.get("market_value") or 0))
        for p in positions
        if str(p.get("symbol") or "").upper()
    }
    aggregate = defaultdict(lambda: {"etf_exposure_usd": 0.0, "funds": set(), "weight_contributions": []})
    funds = []
    for symbol, snapshot in ETF_HOLDINGS.items():
        fund_value = direct.get(symbol, 0.0)
        if fund_value <= 0:
            continue
        rows = []
        for holding, weight in snapshot["holdings"]:
            exposure = fund_value * weight / 100.0
            row = {"symbol": holding, "weight_pct": weight, "exposure_usd": round(exposure, 2)}
            rows.append(row)
            if not holding.endswith("_PRIVATE"):
                item = aggregate[holding]
                item["etf_exposure_usd"] += exposure
                item["funds"].add(symbol)
                item["weight_contributions"].append({"fund": symbol, "weight_pct": weight})
        funds.append({
            "fund": symbol, "market_value_usd": round(fund_value, 2),
            "as_of": snapshot["as_of"], "source": snapshot["source"],
            "top_holdings_coverage_pct": snapshot["coverage_pct"],
            "unattributed_value_usd": round(fund_value * (1 - snapshot["coverage_pct"] / 100), 2),
            "holdings": rows,
        })
    direct_overlap = []
    for symbol, item in aggregate.items():
        direct_value = direct.get(symbol, 0.0)
        etf_value = item["etf_exposure_usd"]
        direct_overlap.append({
            "symbol": symbol, "direct_market_value_usd": round(direct_value, 2),
            "lookthrough_etf_exposure_usd": round(etf_value, 2),
            "combined_economic_exposure_usd": round(direct_value + etf_value, 2),
            "held_via_funds": sorted(item["funds"]),
            "fund_weight_contributions": item["weight_contributions"],
        })
    direct_overlap.sort(key=lambda x: x["combined_economic_exposure_usd"], reverse=True)
    return {
        "methodology": "Issuer-published dated top-holdings snapshots only; not complete holdings. Uncovered fund weight is unattributed. Private/unlisted constituents are shown in fund detail but excluded from listed-stock overlap totals. Fund wrapper market value is not added to underlying exposures a second time.",
        "funds": funds,
        "underlying_overlap": direct_overlap,
        "supported_funds": sorted(f["fund"] for f in funds),
        "unsupported_held_funds": sorted(
            str(p.get("symbol") or "").upper()
            for p in positions
            if str(p.get("instrument_type") or "").lower() in {"etf", "leveraged etf"}
            and str(p.get("symbol") or "").upper() not in ETF_HOLDINGS
            and abs(float(p.get("market_value") or 0)) > 0
        ),
    }
