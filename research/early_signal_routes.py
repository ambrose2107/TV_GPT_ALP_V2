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
_NEWS_BATCH_CACHE = {"at": 0.0, "payload": {}}
_DISCOVERY_CACHE = {"at": 0.0, "symbols": []}
CACHE_SECONDS = 900
NEWS_CACHE_SECONDS = 1800
DISCOVERY_CACHE_SECONDS = 1800
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

def _marketaux_batch_news(symbols, force=False):
    """Fetch news/sentiment for the whole ranked list in one MarketAux request."""
    symbols = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]
    symbols = list(dict.fromkeys(symbols))[:50]
    if not symbols: return {}
    token = _marketaux_token()
    if not token: return {}
    now = time.time()
    cached = _NEWS_BATCH_CACHE.get("payload") or {}
    if cached and not force and now - _NEWS_BATCH_CACHE.get("at", 0.0) < NEWS_CACHE_SECONDS: return cached
    params = {"api_token": token, "symbols": ",".join(symbols), "filter_entities": "true", "must_have_entities": "true", "language": "en", "limit": 50, "published_after": (datetime.now(timezone.utc) - timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M"), "group_similar": "true"}
    try:
        response = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=12)
        response.raise_for_status()
        by_symbol = {s: {"articles": [], "sentiments": []} for s in symbols}
        for item in (response.json().get("data") or []):
            for entity in (item.get("entities") or []):
                symbol = str(entity.get("symbol", "")).upper()
                if symbol not in by_symbol: continue
                score = entity.get("sentiment_score")
                try: score = float(score) if score is not None else None
                except (TypeError, ValueError): score = None
                by_symbol[symbol]["articles"].append({"title": item.get("title") or "Untitled", "source": item.get("source") or "", "published_at": item.get("published_at"), "url": item.get("url"), "snippet": item.get("snippet") or item.get("description") or "", "sentiment_score": round(score, 3) if score is not None else None})
                if score is not None: by_symbol[symbol]["sentiments"].append(score)
        result = {}
        for symbol, bucket in by_symbol.items():
            articles = bucket["articles"][:3]; scores = bucket["sentiments"]
            avg = sum(scores) / len(scores) if scores else None
            result[symbol] = {"configured": True, "symbol": symbol, "articles": articles, "article_count": len(articles), "sentiment_score": round(avg, 3) if avg is not None else None, "sentiment_label": ("Bullish" if avg is not None and avg >= 0.15 else "Bearish" if avg is not None and avg <= -0.15 else "Neutral" if avg is not None else "No signal"), "provider": "MarketAux", "cached": False, "as_of": datetime.now(timezone.utc).isoformat()}
        _NEWS_BATCH_CACHE.update({"at": now, "payload": result})
        return result
    except Exception:
        logger.exception("MarketAux batch news processing failed")
        return {}

def _discover_new_symbols():
    """Discover additional equity symbols from recent financial news."""
    token = _marketaux_token()
    if not token: return []
    now = time.time()
    if _DISCOVERY_CACHE["symbols"] and now - _DISCOVERY_CACHE["at"] < DISCOVERY_CACHE_SECONDS: return list(_DISCOVERY_CACHE["symbols"])
    params = {"api_token": token, "filter_entities": "true", "must_have_entities": "true", "language": "en", "limit": 50, "published_after": (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M"), "group_similar": "true"}
    try:
        response = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=12)
        response.raise_for_status(); found=[]; existing=set(UNIVERSE)
        for item in (response.json().get("data") or []):
            for entity in (item.get("entities") or []):
                symbol=str(entity.get("symbol","")).upper()
                if not symbol or symbol in existing or symbol in found or str(entity.get("type","")).lower()!="equity": continue
                if symbol.replace(".","").replace("-","").isalnum() and len(symbol)<=6: found.append(symbol)
                if len(found)>=25: break
            if len(found)>=25: break
        _DISCOVERY_CACHE.update({"at":now,"symbols":found}); return found
    except Exception:
        logger.exception("MarketAux stock discovery failed"); return []
@early_signal_bp.route("/api/early-signal/research/<symbol>")
def early_signal_research(symbol):
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    symbol = symbol.strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or len(symbol) > 12:
        return jsonify({"error": "Invalid ticker symbol."}), 400
    try:
        from research.ollama_analysis import get_research_confluence_analysis, get_local_ai_analysis
        news = _marketaux_batch_news([symbol], force=False).get(symbol) or _marketaux_news(symbol)
        technical = get_local_ai_analysis(symbol, ["15m", "1h", "1D"])
        synthesis = get_research_confluence_analysis(symbol, technical.get("indicators") or {}, news)
        return jsonify({
            "symbol": symbol,
            "technical_score": technical.get("indicators", {}).get("overall_score"),
            "technical_label": technical.get("indicators", {}).get("overall_label"),
            "news": news,
            "ai_model": synthesis.get("model") or technical.get("model"),
            "analysis": synthesis.get("analysis"),
        })
    except Exception as ex:
        logger.exception("Early-signal research synthesis failed")
        return jsonify({"error": str(ex)}), 500

@early_signal_bp.route("/api/market-news")
def market_news():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    symbol = __import__("flask").request.args.get("symbol", "").strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or len(symbol) > 12:
        return jsonify({"error": "Invalid ticker symbol."}), 400
    return jsonify(_marketaux_news(symbol))

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
    request_obj = __import__("flask").request
    force = request_obj.args.get("refresh") == "1"
    discover = request_obj.args.get("discover") == "1"
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
    scan_universe = list(UNIVERSE)
    discovered_symbols = _discover_new_symbols() if discover else []
    for symbol in discovered_symbols:
        if symbol not in scan_universe: scan_universe.append(symbol)
    params = {
        "symbols": ",".join(scan_universe), "timeframe": "1Day", "start": start,
        "limit": 10000, "adjustment": "split", "feed": "iex", "sort": "asc",
    }
    try:
        response = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=headers,
                                params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        bars_by_symbol = data.get("bars") or {}
        rows, meta = _score_rows(bars_by_symbol)
        news_map = _marketaux_batch_news([r["symbol"] for r in rows[:40]], force=discover)
        for row in rows:
            news = news_map.get(row["symbol"]) or {}
            row["news_sentiment_score"] = news.get("sentiment_score")
            row["news_sentiment"] = news.get("sentiment_label", "No signal")
            row["news_count"] = news.get("article_count", 0)
            row["news_headlines"] = news.get("articles", [])
        if not rows:
            return jsonify({"error": "Alpaca returned no usable daily bars. Check API access and market-data permissions.",
                            "symbols_received": len(bars_by_symbol)}), 502
        payload = {
            "as_of": datetime.now(timezone.utc).isoformat(), "provider": "Alpaca daily bars (IEX feed)",
            "cached": False, "cache_seconds": CACHE_SECONDS, "universe_count": len(scan_universe),
            "symbols_received": len(bars_by_symbol), "candidate_count": len(rows), "discovered_symbols": discovered_symbols, "news_enabled": bool(_marketaux_token()),
            "rows": rows[:40], "meta": meta,
            "methodology": "Technical rank = equal-weight volume expansion, 5-day momentum, moving-average trend, and 5-day relative strength versus SPY. MarketAux news/sentiment is automatically attached to the ranked list in one batched request and cached for 30 minutes. Research context only; not a forecast or trade instruction.",
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
