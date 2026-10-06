"""Automatic Early-Signal stock research scanner.

Uses Alpaca daily bars for a bounded, curated liquid-US-stock universe. No manual
CSV upload is required. Results are cached to protect the Render free tier.
"""
from datetime import datetime, timedelta, timezone
import math
import os
import time

import requests
from flask import Blueprint, jsonify, session

from core.config import Config
from core.logger import get_logger

early_signal_bp = Blueprint("early_signal", __name__)
logger = get_logger(__name__)
_CACHE = {"at": 0.0, "payload": None}
_NEWS_CACHE = {}
CACHE_SECONDS = 900
NEWS_CACHE_SECONDS = 1800
TIMEOUT_SECONDS = 18

# Curated liquid names across broad market, technology, semiconductors, AI
# infrastructure, power, healthcare, financials, consumer and industrials.
# This is a manageable research universe, not a claim to cover every US listing.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "AVGO", "AMD", "TSM", "MU", "AMAT", "LRCX", "KLAC", "ASML", "QCOM", "ARM",
    "MRVL", "ANET", "SMCI", "DELL", "ORCL", "CRM", "NOW", "PLTR", "SNOW", "CRWD",
    "PANW", "TSLA", "NFLX", "UBER", "SHOP", "COIN", "HOOD", "JPM", "GS", "BAC",
    "V", "MA", "COST", "WMT", "LLY", "UNH", "ABBV", "ISRG", "XOM", "CVX",
    "GE", "CAT", "ETN", "VST", "CEG", "FSLR", "NEE", "BE", "LITE", "SNDK",
    "INTC", "TXN", "ADI", "IBM", "CSCO", "ADBE", "DIS", "KO", "TMO", "RTX",
]

def _credentials():
    key = (getattr(Config, "ALPACA_API_KEY", "") or os.environ.get("APCA_API_KEY_ID", "")
           or os.environ.get("ALPACA_KEY", ""))
    secret = (getattr(Config, "ALPACA_SECRET_KEY", "") or os.environ.get("APCA_API_SECRET_KEY", "")
              or os.environ.get("ALPACA_SECRET", ""))
    return key, secret

def _marketaux_token():
    return (os.environ.get("MARKETAUX_API_TOKEN", "")
            or os.environ.get("MARKETAUX_API_KEY", "")
            or "").strip()

def _marketaux_news(symbol):
    """Fetch a tiny, cached news/sentiment feed for one ticker.
    Kept separate from the scanner ranking so the free API quota is not
    consumed on every automatic scan refresh.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"error": "Ticker is required."}
    token = _marketaux_token()
    if not token:
        return {"configured": False, "symbol": symbol, "articles": [],
                "message": "Set MARKETAUX_API_TOKEN in Render to enable news and sentiment."}

    now = time.time()
    cached = _NEWS_CACHE.get(symbol)
    if cached and now - cached["at"] < NEWS_CACHE_SECONDS:
        payload = dict(cached["payload"])
        payload["cached"] = True
        return payload

    params = {
        "api_token": token,
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "limit": 3,
        "published_after": (datetime.now(timezone.utc) - timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M"),
        "group_similar": "true",
    }
    try:
        response = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        raw = response.json()
        articles = []
        sentiments = []
        for item in (raw.get("data") or [])[:3]:
            entities = item.get("entities") or []
            matched = [e for e in entities if str(e.get("symbol", "")).upper() == symbol]
            scores = [float(e.get("sentiment_score")) for e in matched
                      if e.get("sentiment_score") is not None]
            if scores:
                sentiments.extend(scores)
            articles.append({
                "title": item.get("title") or "Untitled",
                "source": item.get("source") or "",
                "published_at": item.get("published_at"),
                "url": item.get("url"),
                "snippet": item.get("snippet") or item.get("description") or "",
                "sentiment_score": round(sum(scores) / len(scores), 3) if scores else None,
            })
        avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
        payload = {
            "configured": True,
            "symbol": symbol,
            "articles": articles,
            "article_count": len(articles),
            "news_intensity": _news_intensity(len(articles)),
            "sentiment_score": round(avg, 3) if sentiments else None,
            "sentiment_label": (
                "Bullish" if avg >= 0.15 else
                "Bearish" if avg <= -0.15 else
                "Neutral"
            ) if sentiments else "No signal",
            "provider": "MarketAux",
            "cached": False,
            "cache_seconds": NEWS_CACHE_SECONDS,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        _NEWS_CACHE[symbol] = {"at": now, "payload": payload}
        return payload
    except requests.HTTPError as ex:
        status = ex.response.status_code if ex.response is not None else 502
        if status == 402:
            return {"configured": True, "symbol": symbol, "articles": [],
                    "error": "MarketAux daily free quota has been reached. Try again tomorrow."}
        logger.warning("MarketAux request rejected: HTTP %s", status)
        return {"configured": True, "symbol": symbol, "articles": [],
                "error": "MarketAux rejected the news request (HTTP %s)." % status}
    except requests.RequestException as ex:
        logger.warning("MarketAux news unavailable: %s", ex)
        return {"configured": True, "symbol": symbol, "articles": [],
                "error": "MarketAux news is temporarily unavailable."}
    except Exception:
        logger.exception("MarketAux news processing failed")
        return {"configured": True, "symbol": symbol, "articles": [],
                "error": "News/sentiment processing failed."}


@early_signal_bp.route("/api/early-signal/research/<symbol>")
def early_signal_research(symbol):
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    symbol = (symbol or "").strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or len(symbol) > 12:
        return jsonify({"error": "Invalid ticker symbol."}), 400
    scan = early_signal_scan().get_json()
    if not scan or not scan.get("rows"):
        return jsonify({"error": "Technical scanner data is unavailable."}), 503
    row = next((r for r in scan["rows"] if r.get("symbol") == symbol), None)
    if row is None:
        return jsonify({"error": "Ticker is not in the current ranked research universe."}), 404
    news = _marketaux_news(symbol)
    news_score = news.get("sentiment_score")
    vc, vw = _volume_confirmation(row.get("return_5d_pct"), row.get("relative_volume"))
    state, rationale = _confluence_state(row.get("score"), news_score, vc)
    return jsonify({
        "symbol": symbol,
        "technical": row,
        "news": news,
        "volume_confirmation": vc,
        "volume_warning": vw,
        "news_intensity": news.get("news_intensity", "None"),
        "overall_state": state,
        "rationale": rationale,
        "methodology": "Research confluence combines technical score, MarketAux news sentiment/intensity, and deterministic price-volume confirmation. It is not a forecast or trade instruction."
    })

@early_signal_bp.route("/api/market-news")
def market_news():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    symbol = __import__("flask").request.args.get("symbol", "").strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or len(symbol) > 12:
        return jsonify({"error": "Invalid ticker symbol."}), 400
    return jsonify(_marketaux_news(symbol))


def _news_intensity(article_count):
    n = int(article_count or 0)
    return "High" if n >= 3 else ("Medium" if n == 2 else ("Low" if n == 1 else "None"))

def _volume_confirmation(return_5d_pct, relative_volume):
    ret = float(return_5d_pct or 0)
    rv = float(relative_volume or 0)
    if ret >= 1.0 and rv >= 1.2:
        return "Strong", "Price momentum is supported by above-average volume."
    if ret >= 1.0 and rv < 1.0:
        return "Weak", "Positive price momentum with below-average volume; divergence warning."
    if ret <= -1.0 and rv >= 1.2:
        return "Strong selling", "Downside momentum is supported by above-average volume."
    if ret <= -1.0 and rv < 1.0:
        return "Weak selling", "Downside move lacks strong volume participation."
    return "Neutral", "No strong price/volume confirmation signal."

def _confluence_state(technical_score, news_score, volume_confirmation):
    t = float(technical_score or 0)
    n = float(news_score or 0)
    if n >= 0.15 and t >= 65 and volume_confirmation == "Strong":
        return "BULLISH", "Technical trend, positive news and volume are aligned."
    if n <= -0.15 and t < 45 and volume_confirmation in ("Strong selling", "Weak selling"):
        return "BEARISH", "Technical weakness and negative news are aligned."
    if t >= 65 and n < -0.15:
        return "MIXED", "Technicals are strong but news sentiment is negative."
    if t >= 65 and volume_confirmation == "Weak":
        return "BULLISH WITH DIVERGENCE", "Technical momentum is positive, but volume confirmation is weak."
    if t < 45 and n >= 0.15:
        return "MIXED", "News is positive but technical structure is not confirming it."
    return "NEUTRAL", "Signals are not sufficiently aligned for a strong research state."

def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))

def _score_rows(bars_by_symbol):
    spy = bars_by_symbol.get("SPY") or []
    spy_closes = [float(b.get("c", 0) or 0) for b in spy if float(b.get("c", 0) or 0) > 0]
    spy_5d = (spy_closes[-1] / spy_closes[-6] - 1) * 100 if len(spy_closes) >= 6 else 0.0
    rows = []
    skipped = 0
    for symbol, bars in bars_by_symbol.items():
        if symbol in ("SPY", "QQQ", "IWM", "DIA") or len(bars) < 22:
            continue
        try:
            closes = [float(b.get("c", 0) or 0) for b in bars]
            volumes = [float(b.get("v", 0) or 0) for b in bars]
            if any(c <= 0 for c in closes[-21:]):
                skipped += 1
                continue
            last = closes[-1]
            prev5 = closes[-6]
            prev20 = closes[-21]
            avg_vol = sum(volumes[-21:-1]) / max(1, len(volumes[-21:-1]))
            rel_vol = volumes[-1] / avg_vol if avg_vol > 0 else 0.0
            ret5 = (last / prev5 - 1) * 100
            ret20 = (last / prev20 - 1) * 100
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            avg_dollar_vol = sum(closes[i] * volumes[i] for i in range(max(0, len(closes)-20), len(closes))) / min(20, len(closes))
            volume_score = _clamp((rel_vol - 0.8) / 2.2 * 100)
            momentum_score = _clamp((ret5 + 2) / 12 * 100)
            trend_score = (50 if last > sma20 else 0) + (50 if sma50 is not None and sma20 > sma50 else (25 if sma50 is None and last > sma20 else 0))
            rel_strength = ret5 - spy_5d
            relative_score = _clamp((rel_strength + 5) / 10 * 100)
            total = round((volume_score + momentum_score + trend_score + relative_score) / 4, 1)
            rows.append({
                "symbol": symbol, "price": round(last, 2), "day_change_pct": round((last / closes[-2] - 1) * 100, 2) if len(closes) > 1 else None,
                "return_5d_pct": round(ret5, 2), "return_20d_pct": round(ret20, 2),
                "relative_volume": round(rel_vol, 2), "avg_dollar_volume": round(avg_dollar_vol, 0),
                "volume_score": round(volume_score, 1), "momentum_score": round(momentum_score, 1),
                "trend_score": round(trend_score, 1), "relative_strength_score": round(relative_score, 1),
                "relative_strength_vs_spy_pct": round(rel_strength, 2), "score": total,
                "trend": "Uptrend" if last > sma20 and sma50 is not None and sma20 > sma50 else ("Above 20D" if last > sma20 else "Mixed/weak"),
                "volume_confirmation": _volume_confirmation(ret5, rel_vol)[0],
                "volume_warning": _volume_confirmation(ret5, rel_vol)[1] if _volume_confirmation(ret5, rel_vol)[0] == "Weak" else "",
                "bars": len(bars),
            })
        except (TypeError, ValueError, ZeroDivisionError):
            skipped += 1
    rows.sort(key=lambda r: (r["score"], r["avg_dollar_volume"]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row["signal"] = "Strong setup to review" if row["score"] >= 75 else ("Watchlist" if row["score"] >= 60 else "Developing")
    return rows, {"benchmark": "SPY", "benchmark_return_5d_pct": round(spy_5d, 2), "skipped_symbols": skipped}

@early_signal_bp.route("/api/early-signal/scan")
def early_signal_scan():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    force = __import__("flask").request.args.get("refresh") == "1"
    now = time.time()
    if not force and _CACHE["payload"] is not None and now - _CACHE["at"] < CACHE_SECONDS:
        payload = dict(_CACHE["payload"])
        payload["cached"] = True
        return jsonify(payload)
    key, secret = _credentials()
    if not key or not secret:
        return jsonify({"error": "Automatic scanner needs the existing Alpaca API key and secret configured on the server."}), 503
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    start = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat().replace("+00:00", "Z")
    params = {
        "symbols": ",".join(UNIVERSE), "timeframe": "1Day", "start": start,
        "limit": 10000, "adjustment": "split", "feed": "iex", "sort": "asc",
    }
    try:
        response = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=headers,
                                params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        bars_by_symbol = data.get("bars") or {}
        rows, meta = _score_rows(bars_by_symbol)
        if not rows:
            return jsonify({"error": "Alpaca returned no usable daily bars. Check API access and market-data permissions.",
                            "symbols_received": len(bars_by_symbol)}), 502
        payload = {
            "as_of": datetime.now(timezone.utc).isoformat(), "provider": "Alpaca daily bars (IEX feed)",
            "cached": False, "cache_seconds": CACHE_SECONDS, "universe_count": len(UNIVERSE),
            "symbols_received": len(bars_by_symbol), "candidate_count": len(rows),
            "rows": rows[:40], "meta": meta,
            "methodology": "Score = equal-weight volume expansion, 5-day momentum, moving-average trend, and 5-day relative strength versus SPY. MarketAux news/sentiment is available on demand; it is intentionally not included in the automatic rank because the free plan is quota-limited. Research ranking only; not a forecast or trade instruction.",
        }
        _CACHE.update({"at": now, "payload": payload})
        return jsonify(payload)
    except requests.HTTPError as ex:
        status = ex.response.status_code if ex.response is not None else 502
        detail = "Alpaca rejected the market-data request (HTTP %s). Check credentials, plan/feed access, and API status." % status
        logger.warning("Early-signal Alpaca request rejected: %s", status)
        return jsonify({"error": detail}), 502
    except requests.RequestException as ex:
        logger.warning("Early-signal market data unavailable: %s", ex)
        return jsonify({"error": "Automatic market data is temporarily unavailable. Try refresh later."}), 503
    except Exception:
        logger.exception("Early-signal scanner failed")
        return jsonify({"error": "Scanner failed while processing market data."}), 500
