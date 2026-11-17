"""Static, reviewable classification of portfolio instruments.

Primary sectors follow broad GICS-style issuer classification. Themes describe
economic exposure and are intentionally separate from issuer sector. ETF
look-through is not assumed unless a current holdings feed is available.
"""
from collections import defaultdict

# ticker: (sector, industry, primary_theme, instrument_type, description, risk_tags)
CLASSIFICATIONS = {
    "AAOI": ("Information Technology", "Optical components", "AI data-centre networking", "Equity", "Optical transceivers and networking components", ["AI capex", "semiconductor supply chain"]),
    "ADBE": ("Information Technology", "Application software", "Enterprise software and digital media", "Equity", "Creative and document software", ["software spending", "AI monetisation"]),
    "AMD": ("Information Technology", "Semiconductors", "AI accelerators and CPUs", "Equity", "Processors and data-centre accelerators", ["AI capex", "semiconductor cycle"]),
    "AMZN": ("Consumer Discretionary", "Broadline retail and cloud", "E-commerce and cloud infrastructure", "Equity", "Online retail and AWS cloud services", ["consumer demand", "cloud capex"]),
    "ASML": ("Information Technology", "Semiconductor equipment", "Semiconductor manufacturing equipment", "Equity", "Lithography systems for chip manufacturing", ["chip capex", "export controls"]),
    "BE": ("Industrials", "Electrical equipment", "Data-centre and distributed power", "Equity", "Fuel-cell power systems", ["project execution", "financing", "power demand"]),
    "CCJ": ("Energy", "Uranium", "Nuclear fuel and uranium", "Equity", "Uranium mining and nuclear fuel services", ["uranium price", "project execution"]),
    "CIBR": ("ETF / Fund", "Cybersecurity ETF", "Cybersecurity", "ETF", "Diversified cybersecurity companies", ["equity market", "fund overlap"]),
    "COIN": ("Financials", "Capital markets / crypto platform", "Crypto exchange and digital assets", "Equity", "Crypto trading platform and services", ["crypto prices", "regulation", "trading volumes"]),
    "CONL": ("ETF / Fund", "Single-stock leveraged ETF", "Leveraged Coinbase exposure", "Leveraged ETF", "Daily leveraged exposure to Coinbase; compounding and reset risk", ["daily leverage", "volatility decay", "COIN concentration"]),
    "CRCL": ("Financials", "Financial technology", "Stablecoins and payments infrastructure", "Equity", "Stablecoin issuer and payments infrastructure", ["stablecoin adoption", "rates", "regulation"]),
    "CRWD": ("Information Technology", "Cybersecurity software", "Cybersecurity", "Equity", "Cloud endpoint and security software", ["IT budgets", "competition", "valuation"]),
    "CRWV": ("Information Technology", "Cloud infrastructure", "AI cloud and GPU infrastructure", "Equity", "GPU cloud infrastructure provider", ["AI capex", "customer concentration", "funding"]),
    "FSLR": ("Information Technology", "Solar technology", "Solar manufacturing and energy transition", "Equity", "Thin-film solar module manufacturer", ["policy", "solar pricing", "project demand"]),
    "GOOG": ("Communication Services", "Interactive media and services", "Digital advertising and AI platforms", "Equity", "Search, advertising, cloud and AI platform", ["ad cycle", "AI competition", "regulation"]),
    "IBIT": ("ETF / Fund", "Spot Bitcoin ETF", "Bitcoin", "ETF", "Spot Bitcoin exposure through an exchange-traded fund", ["bitcoin price", "custody"]),
    "IBM": ("Information Technology", "IT services and software", "Enterprise technology and quantum", "Equity", "Enterprise IT, hybrid cloud and quantum research", ["enterprise spending", "execution"]),
    "INTC": ("Information Technology", "Semiconductors", "Foundry and CPUs", "Equity", "CPU and semiconductor manufacturing", ["manufacturing yields", "capital intensity"]),
    "IONQ": ("Information Technology", "Quantum computing", "Quantum computing", "Equity", "Trapped-ion quantum computing systems", ["commercialisation", "funding", "valuation"]),
    "IREN": ("Information Technology", "Data-centre infrastructure", "AI data centres and Bitcoin mining", "Equity", "Data-centre operator with AI cloud and Bitcoin mining exposure", ["power availability", "AI capex", "bitcoin price"]),
    "JNJ": ("Health Care", "Pharmaceuticals and medical technology", "Healthcare and defensive", "Equity", "Diversified pharmaceuticals and medical technology", ["drug pipeline", "litigation"]),
    "KLAC": ("Information Technology", "Semiconductor equipment", "Chip process control", "Equity", "Semiconductor process-control and inspection equipment", ["chip capex", "export controls"]),
    "LITE": ("Information Technology", "Optical components", "AI data-centre networking", "Equity", "Optical and photonics components", ["AI capex", "customer concentration"]),
    "LRCX": ("Information Technology", "Semiconductor equipment", "Wafer fabrication equipment", "Equity", "Wafer fabrication and etch equipment", ["chip capex", "export controls"]),
    "MRVL": ("Information Technology", "Semiconductors", "AI networking and custom silicon", "Equity", "Data-centre networking and custom silicon", ["AI capex", "customer concentration"]),
    "MU": ("Information Technology", "Semiconductors", "Memory and HBM", "Equity", "DRAM, NAND and high-bandwidth memory", ["memory pricing", "capex cycle"]),
    "NBIS": ("Information Technology", "Cloud infrastructure", "AI cloud and GPU infrastructure", "Equity", "AI-focused cloud and GPU infrastructure", ["AI capex", "funding", "execution"]),
    "NOK": ("Information Technology", "Communications equipment", "Telecom networks and optical infrastructure", "Equity", "Telecom network and optical networking equipment", ["carrier capex", "competition"]),
    "NOW": ("Information Technology", "Application software", "Enterprise workflow software", "Equity", "Enterprise workflow and automation software", ["IT budgets", "AI monetisation"]),
    "NVDA": ("Information Technology", "Semiconductors", "AI accelerators and platforms", "Equity", "GPUs, networking and accelerated computing platform", ["AI capex", "export controls", "customer concentration"]),
    "NXT": ("Industrials", "Electrical equipment", "Solar infrastructure", "Equity", "Solar tracking systems", ["solar project demand", "policy"]),
    "ONDS": ("Industrials", "Aerospace and communications equipment", "Drones and private wireless networks", "Equity", "Autonomous drones and industrial wireless networks", ["contract execution", "funding"]),
    "ORCL": ("Information Technology", "Software and cloud", "Cloud infrastructure and enterprise software", "Equity", "Enterprise software and cloud infrastructure", ["cloud capex", "debt and capital intensity"]),
    "OUST": ("Information Technology", "Electronic equipment", "Lidar sensing and autonomy", "Equity", "Digital lidar sensors and perception software", ["industrial adoption", "competition"]),
    "PANW": ("Information Technology", "Cybersecurity software", "Cybersecurity", "Equity", "Network, cloud and security platform", ["IT budgets", "competition"]),
    "QBTS": ("Information Technology", "Quantum computing", "Quantum computing", "Equity", "Quantum annealing systems", ["commercialisation", "funding", "valuation"]),
    "QQQ": ("ETF / Fund", "Nasdaq-100 ETF", "Large-cap growth and technology", "ETF", "Nasdaq-100 exposure; technology-heavy and concentrated", ["mega-cap overlap", "equity market"]),
    "QQQI": ("ETF / Fund", "Nasdaq-100 option-income ETF", "Large-cap growth with option income", "ETF", "Nasdaq exposure with an option-income strategy", ["equity market", "option upside cap", "distribution variability"]),
    "QUBT": ("Information Technology", "Quantum and photonics technology", "Quantum computing and photonics", "Equity", "Quantum optics and photonics technology", ["commercialisation", "funding", "valuation"]),
    "QTEX": ("Health Care", "Medical devices and advanced electronics", "Medical technology and quantum connectivity", "Equity", "Medical devices plus developing advanced connectivity for quantum systems", ["commercialisation", "funding", "execution"]),
    "RGTI": ("Information Technology", "Quantum computing", "Quantum computing", "Equity", "Superconducting quantum processors", ["commercialisation", "funding", "valuation"]),
    "RKLB": ("Industrials", "Aerospace and defense", "Space launch and satellite systems", "Equity", "Launch services and space systems", ["launch execution", "capital intensity"]),
    "RR": ("Industrials", "Machinery and robotics", "Robotics and automation", "Equity", "Robotic systems for commercial and industrial applications", ["adoption", "execution", "funding"]),
    "S": ("Information Technology", "Cybersecurity software", "Cybersecurity", "Equity", "Endpoint and cloud security software", ["IT budgets", "competition"]),
    "SHAZ": ("Information Technology", "AI infrastructure", "AI compute and data-centre infrastructure", "Equity", "AI and high-performance-computing infrastructure exposure", ["AI capex", "funding", "small-cap volatility"]),
    "SMH": ("ETF / Fund", "Semiconductor ETF", "Semiconductors and equipment", "ETF", "Semiconductor-sector ETF", ["chip-cycle concentration", "holdings overlap"]),
    "SNDK": ("Information Technology", "Semiconductors and storage", "NAND flash and data storage", "Equity", "Flash memory and data-storage products", ["NAND pricing", "inventory cycle"]),
    "SOFI": ("Financials", "Consumer finance and fintech", "Digital banking and fintech", "Equity", "Digital bank and consumer-finance platform", ["credit quality", "funding costs", "regulation"]),
    "SOXX": ("ETF / Fund", "Semiconductor ETF", "Semiconductors and equipment", "ETF", "Semiconductor-sector ETF", ["chip-cycle concentration", "holdings overlap"]),
    "TSLA": ("Consumer Discretionary", "Automobiles", "Electric vehicles, autonomy and energy", "Equity", "Electric vehicles, software and energy storage", ["auto demand", "pricing", "execution"]),
    "TSM": ("Information Technology", "Semiconductors", "Foundry and advanced packaging", "Equity", "Leading contract semiconductor foundry", ["geopolitics", "chip capex", "customer concentration"]),
    "VRT": ("Industrials", "Electrical equipment", "Data-centre power and cooling", "Equity", "Power, thermal management and infrastructure for data centres", ["AI capex", "power constraints", "execution"]),
    "VST": ("Utilities", "Electric utilities", "Power generation and data-centre demand", "Equity", "Competitive electricity generation and retail power", ["power prices", "regulation", "fuel costs"]),
    "WQTM": ("ETF / Fund", "Quantum computing ETF", "Quantum computing", "ETF", "Global quantum-computing thematic equity ETF", ["theme volatility", "holdings overlap"]),
    "XLU": ("ETF / Fund", "Utilities sector ETF", "Utilities", "ETF", "US utilities-sector ETF", ["rates", "regulated returns", "fund overlap"]),
    "ZS": ("Information Technology", "Cybersecurity software", "Cybersecurity", "Equity", "Cloud-native network and zero-trust security", ["IT budgets", "competition"]),
    "SOLUSD": ("Crypto", "Digital asset", "Solana ecosystem", "Crypto", "Direct Solana token exposure", ["crypto volatility", "network and regulatory risk"]),
}

def classify_position(position):
    symbol = str(position.get("symbol") or "").upper()
    row = CLASSIFICATIONS.get(symbol)
    if not row:
        return {
            "sector": "Unclassified", "industry": "Needs review",
            "theme": "Unclassified", "instrument_type": "Unknown",
            "description": "No verified classification mapping; excluded from sector conclusions until reviewed.",
            "risk_tags": ["classification gap"], "classification_status": "unmapped",
        }
    sector, industry, theme, instrument_type, description, risk_tags = row
    return {
        "sector": sector, "industry": industry, "theme": theme,
        "instrument_type": instrument_type, "description": description,
        "risk_tags": risk_tags, "classification_status": "mapped",
    }

def enrich_portfolio(positions, account_equity=0.0):
    """Return enriched positions plus non-look-through sector/theme allocations."""
    rows = []
    gross = sum(abs(float(p.get("market_value") or 0)) for p in positions)
    for original in positions:
        p = dict(original)
        p.update(classify_position(p))
        mv = abs(float(p.get("market_value") or 0))
        p["gross_weight_pct"] = (mv / gross * 100) if gross else 0.0
        p["equity_weight_pct"] = (mv / account_equity * 100) if account_equity else 0.0
        rows.append(p)

    def aggregate(field):
        buckets = defaultdict(lambda: {"market_value": 0.0, "symbols": [], "position_count": 0})
        for p in rows:
            key = p.get(field) or "Unclassified"
            bucket = buckets[key]
            bucket["market_value"] += abs(float(p.get("market_value") or 0))
            bucket["symbols"].append(p["symbol"])
            bucket["position_count"] += 1
        result = []
        for key, value in buckets.items():
            mv = value["market_value"]
            result.append({
                field: key, "market_value": round(mv, 2),
                "gross_weight_pct": round(mv / gross * 100, 2) if gross else 0.0,
                "equity_weight_pct": round(mv / account_equity * 100, 2) if account_equity else 0.0,
                "weight_pct": round(mv / account_equity * 100, 2) if account_equity else 0.0,
                "position_count": value["position_count"],
                "symbols": value["symbols"],
            })
        return sorted(result, key=lambda x: -x["market_value"])

    return {
        "positions": rows,
        "sector_exposure": aggregate("sector"),
        "theme_exposure_detailed": aggregate("theme"),
        "instrument_exposure": aggregate("instrument_type"),
        "classification_coverage_pct": round(
            sum(1 for p in rows if p["classification_status"] == "mapped") / len(rows) * 100, 1
        ) if rows else 100.0,
        "classification_unmapped": [p["symbol"] for p in rows if p["classification_status"] != "mapped"],
        "gross_market_value": round(gross, 2),
        "methodology": "Issuer sector and primary investment theme are mutually exclusive per position. ETFs are classified as fund wrappers and are not look-through attributed to underlying sectors unless holdings data is supplied; this prevents double counting. Exposure uses absolute current market value, not position count.",
    }
