"""
research/ai_research.py
All research powered by Claude + web_search.
Works on Railway (yfinance also available there).
"""
import requests
import os
from core.logger import get_logger

logger = get_logger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

def _call_claude(prompt: str, use_search: bool = True, max_tokens: int = 1500) -> str:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14" if False else "",
    }
    headers = {k:v for k,v in headers.items() if v}
    
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    try:
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]
        return text.strip()
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return f"Error fetching analysis: {str(e)}"


def get_institutional_analysis() -> dict:
    prompt = """You are a financial research analyst. Search for and analyze the latest 13F SEC filings for:
    Berkshire Hathaway, Bridgewater Associates, Renaissance Technologies, Citadel, and Two Sigma.

    For each fund identify: top 10 holdings, positions increased most in last 90 days, new positions added, sector concentration.

    Then identify TOP 5 STOCKS where institutional buying has accelerated across multiple funds in the last 90 days but retail attention is still low.
    For each stock give: estimated average entry price range, current approximate share count across funds, underlying thesis, why smart money is loading up before the crowd notices.

    Format as a clear research report with sections per fund then the consolidated top 5 picks with a conviction table."""

    text = _call_claude(prompt, use_search=True, max_tokens=2000)
    return {"analysis": text, "type": "institutional"}


def get_earnings_whiplash() -> dict:
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. You are a volatility arbitrage analyst.

    For the next 14 days of S&P 500 earnings reports:
    1. Identify 10 stocks with highest historical post-earnings volatility (8%+ moves historically)
    2. For each: historical average post-earnings move %, current implied volatility from options, whether IV is lower than historical HV
    3. FLAG the 3 names where implied volatility is meaningfully LOWER than historical realized volatility — these are asymmetric setups
    4. For each flagged stock: expected move from IV, historical move delivered, the mispricing gap, opportunity explanation

    Use real current market data. Present as a trading research report with a clear ASYMMETRIC SETUPS section and a TOP PICKS table."""

    text = _call_claude(prompt, use_search=True, max_tokens=1800)
    return {"analysis": text, "type": "earnings"}


def get_sector_rotation() -> dict:
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. You are a sector rotation analyst.

    PART A: Compare last 30 days of S&P 500 sector performance (all 11 GICS sectors: XLK, XLV, XLF, XLY, XLI, XLC, XLP, XLE, XLB, XLU, XLRE) against the same period 1 year ago.
    Show exact % returns for both periods.

    PART B: Identify sectors showing LEADERSHIP ROTATION — relative strength flipping from negative to positive or positive to negative.

    PART C: For each rotating sector, pull the 5 highest-volume ETFs and rank by money flow over last 10 and 30 days.

    PART D: Give 3 most actionable sector rotation trades — sector, best ETF, entry thesis, price target, risk. Confidence level for each.

    Format with clear sections. End with a TRADES TABLE."""

    text = _call_claude(prompt, use_search=True, max_tokens=2000)
    return {"analysis": text, "type": "sectors"}


def get_insider_confluence(symbols=None) -> dict:
    if not symbols:
        symbols = "AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,JPM,BAC,AMZN,NFLX,CRM,ORCL,INTC,MU"
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. You are an insider trading and options flow analyst.

    PART A — SEC Form 4 Insider Purchases (last 30 days):
    Find recent Form 4 filings showing insider OPEN MARKET PURCHASES (not grants) over $500,000.
    For each: executive name/title, company/ticker, dollar amount, date, why it matters.

    PART B — Unusual Options Activity for: {symbols}
    Identify: volume/OI ratio >2x, large call sweeps suggesting directional bets, near-term options with unusual volume.

    PART C — Confluence (TOP 4):
    Find 4 stocks where BOTH insider buying AND unusual call options are occurring simultaneously.
    For each: insider details, options flow details, signal strength, potential catalyst, historical precedent.

    End with a CONVICTION TABLE ranking the 4 confluence stocks (High/Medium/Low confidence)."""

    text = _call_claude(prompt, use_search=True, max_tokens=2000)
    return {"analysis": text, "type": "insider"}


def analyze_stocks_ai(symbols: list, timeframes: list) -> dict:
    """AI-powered multi-stock technical analysis."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    sym_str = ", ".join(symbols)
    tf_str  = ", ".join(timeframes)

    prompt = f"""Today is {today}. You are a technical analysis expert replicating TradingView Analyzer Pro.

    Analyze these stocks: {sym_str}
    Timeframes: {tf_str}

    For EACH stock and EACH timeframe calculate using REAL current market data:
    - RSI(14): value and label (Overbought >70, Neutral 45-70, Oversold <30)
    - MACD: label (Strong Bull/Bull/Neutral/Bear/Strong Bear)
    - ADX(14): value and label (Strong >30, Trending 20-30, No Trend <20 — add Up/Down)
    - Bollinger Bands: label (Upper Break/Above Mid/In Bands/Below Mid/Lower Break)
    - EMA50: % above/below and label (Uptrend/Consolidating/Downtrend)
    - VWAP: % above/below and label (Over/Near/Under)
    - Overall: timeframe signal (Strong Bull/Bull/Neutral/Bear/Strong Bear)

    Return ONLY valid JSON, no markdown, no explanation:
    {{"results":[{{"symbol":"AAPL","price":182.5,"overall":"Bull","tfs":{{"1D":{{"rsi":{{"v":58.2,"l":"Neutral"}},"macd":{{"l":"Bull"}},"adx":{{"v":28.1,"l":"Trending Up"}},"bb":{{"l":"Above Mid"}},"ema50":{{"v":"+2.1%","l":"Uptrend"}},"vwap":{{"v":"+0.8%","l":"Over"}},"result":{{"l":"Bull","css":"bull"}}}}}}}}]}}"""

    text = _call_claude(prompt, use_search=True, max_tokens=2000)
    try:
        import json, re
        clean = re.sub(r'```json|```', '', text).strip()
        data  = json.loads(clean)
        return data
    except:
        return {"raw": text, "error": "Could not parse JSON — returning raw analysis"}
