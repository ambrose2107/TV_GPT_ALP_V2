"""Portfolio Manager: lightweight portfolio risk and profit-taking research."""
from flask import Blueprint, jsonify, render_template, request, session
from core.config import Config
from core.logger import get_logger
import os, math, time
import requests
from brokers.alpaca_adapter import AlpacaAdapter
from datetime import datetime, timezone

portfolio_bp = Blueprint("portfolio_manager", __name__)
logger = get_logger(__name__)
_CACHE = {"at": 0, "payload": None}
CACHE_SECONDS = 300

def _auth_json():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None

def _headers():
    return {
        "APCA-API-KEY-ID": Config.ALPACA_API_KEY or os.environ.get("APCA_API_KEY_ID", "") or os.environ.get("ALPACA_KEY", ""),
        "APCA-API-SECRET-KEY": Config.ALPACA_SECRET_KEY or os.environ.get("APCA_API_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET", ""),
    }

def _safe_float(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default

@portfolio_bp.route("/portfolio-manager")
def portfolio_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("portfolio_manager.html")

@portfolio_bp.route("/api/portfolio-manager")
def portfolio_data():
    err = _auth_json()
    if err: return err
    force = request.args.get("refresh") == "1"
    now = time.time()
    if not force and _CACHE["payload"] and now - _CACHE["at"] < CACHE_SECONDS:
        return jsonify(_CACHE["payload"])
    try:
        # Reuse the same Alpaca adapter already used by the working Positions/Account dashboard.
        # This keeps Portfolio Manager on the exact same credentials, ALPACA_MODE and request path.
        adapter = AlpacaAdapter()
        try:
            account = adapter.get_account()
            raw = adapter.get_positions()
        except Exception as ex:
            # The dashboard's working Positions tab also supports legacy Render
            # environment names.  Retry with those exact credentials before
            # declaring Portfolio Manager unavailable.
            key = Config.ALPACA_API_KEY or os.environ.get("APCA_API_KEY_ID", "") or os.environ.get("ALPACA_KEY", "")
            secret = Config.ALPACA_SECRET_KEY or os.environ.get("APCA_API_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET", "")
            base = Config.ALPACA_BASE_URL
            if not key or not secret:
                raise RuntimeError(
                    "Alpaca credentials are not available to Portfolio Manager. "
                    "The Positions tab may be using legacy APCA_API_KEY_ID/APCA_API_SECRET_KEY names."
                ) from ex
            headers = {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Content-Type": "application/json",
            }
            try:
                ar = requests.get(base + "/v2/account", headers=headers, timeout=8)
                ar.raise_for_status()
                pr = requests.get(base + "/v2/positions", headers=headers, timeout=8)
                pr.raise_for_status()
                account = ar.json()
                raw = pr.json()
            except Exception as retry_ex:
                raise RuntimeError(
                    "Alpaca account/positions request failed. "
                    "Adapter: " + str(ex)[:180] + "; fallback: " + str(retry_ex)[:280]
                ) from retry_ex
        if not isinstance(account, dict):
            raise RuntimeError("Alpaca account response was not an object.")
        if not isinstance(raw, list):
            raise RuntimeError("Alpaca positions response was not a list.")
        equity = _safe_float(account.get("equity") or account.get("portfolio_value"))
        cash = _safe_float(account.get("cash"))
        positions = []
        for p in raw:
            mv = _safe_float(p.get("market_value"))
            cb = _safe_float(p.get("cost_basis"))
            pl = _safe_float(p.get("unrealized_pl"))
            pct = _safe_float(p.get("unrealized_plpc")) * 100
            positions.append({
                "symbol": str(p.get("symbol", "")).upper(), "qty": _safe_float(p.get("qty")),
                "side": p.get("side", "long"), "price": _safe_float(p.get("current_price")),
                "avg_entry": _safe_float(p.get("avg_entry_price")), "market_value": mv,
                "cost_basis": cb, "unrealized_pl": pl, "unrealized_pct": pct,
                "weight_pct": (mv / equity * 100) if equity else 0,
            })
        theme_map = {
            "Semiconductors & equipment": {"AMD","ASML","MRVL","TSM","SNDK","KLAC","MU","INTC","SMH","QCOM","NVDA","AVGO","AMAT","LRCX","SOXX"},
            "AI infrastructure / optical": {"LITE","AAOI","NBIS","CRWV","ORCL","VRT","ANET","DELL","SMCI"},
            "Quantum / emerging technology": {"QBTS","RGTI","IONQ","QUBT"},
            "Space / frontier growth": {"RKLB","ASTS","ONDS","OUST"},
            "Crypto-linked": {"IBIT","CONL","COIN","MARA","RIOT"},
            "Broad market / diversified ETFs": {"SPY","QQQ","IWM","DIA","VTI","VOO","SMH","SOXX"},
        }
        for p in positions:
            p["theme"] = next((name for name, members in theme_map.items() if p["symbol"] in members), "Other / unclassified")
        positions.sort(key=lambda x: abs(x["market_value"]), reverse=True)
        total_mv = sum(abs(p["market_value"]) for p in positions)
        theme_totals = {}
        for p in positions:
            row = theme_totals.setdefault(p["theme"], {"market_value": 0.0, "symbols": []})
            row["market_value"] += abs(p["market_value"])
            row["symbols"].append(p["symbol"])
        theme_exposure = [{"theme": k, "market_value": round(v["market_value"], 2),
                           "weight_pct": (v["market_value"] / equity * 100) if equity else 0,
                           "symbols": v["symbols"]}
                          for k, v in sorted(theme_totals.items(), key=lambda kv: -kv[1]["market_value"])]
        invested = (total_mv / equity * 100) if equity else 0
        # Use Alpaca's market-data API instead of yfinance.  Render often gets
        # Yahoo Finance rate-limit/JSON failures, which can make every ticker fail.
        # One bounded batch request is much lighter and uses the same credentials.
        symbols = list(dict.fromkeys([p["symbol"] for p in positions if p["symbol"]]))[:35]
        market = {}
        warnings = []
        try:
            key = Config.ALPACA_API_KEY or os.environ.get("APCA_API_KEY_ID", "") or os.environ.get("ALPACA_KEY", "")
            secret = Config.ALPACA_SECRET_KEY or os.environ.get("APCA_API_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET", "")
            data_symbols = list(dict.fromkeys(symbols + ["SPY", "QQQ", "IWM", "TLT", "GLD"]))
            if not key or not secret:
                raise RuntimeError("Alpaca market-data credentials are unavailable.")
            from datetime import timedelta
            start_date = (datetime.now(timezone.utc) - timedelta(days=430)).date().isoformat()
            data_url = "https://data.alpaca.markets/v2/stocks/bars"
            # Alpaca caps each page at 10,000 bars across symbols. Follow a
            # small bounded number of page tokens so 12M returns aren't silently
            # truncated when the portfolio contains many tickers.
            bars = {}
            page_token = None
            for _page in range(4):
                params = {
                    "symbols": ",".join(data_symbols),
                    "timeframe": "1Day",
                    "start": start_date,
                    "limit": 10000,
                    "adjustment": "all",
                    "feed": "iex",
                    "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token
                resp = requests.get(
                    data_url, params=params,
                    headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                    timeout=12,
                )
                resp.raise_for_status()
                body = resp.json()
                page_bars = body.get("bars") if isinstance(body, dict) else None
                if not isinstance(page_bars, dict):
                    raise ValueError("Alpaca market-data response did not contain bars.")
                for sym, rows in page_bars.items():
                    if isinstance(rows, list):
                        bars.setdefault(sym, []).extend(rows)
                page_token = body.get("next_page_token")
                if not page_token:
                    break
            import pandas as pd
            close_data = {}
            high_data = {}
            for sym, rows in bars.items():
                if not isinstance(rows, list):
                    continue
                vals = {}
                for row in rows:
                    try:
                        ts = row.get("t")
                        close_px = float(row.get("c"))
                        if ts and math.isfinite(close_px):
                            vals[ts] = close_px
                            try:
                                high_px = float(row.get("h"))
                                if math.isfinite(high_px):
                                    high_data.setdefault(sym, {})[ts] = high_px
                            except (TypeError, ValueError):
                                pass
                    except (TypeError, ValueError):
                        continue
                if vals:
                    close_data[sym] = pd.Series(vals, dtype="float64")
            close = pd.DataFrame(close_data).sort_index()
            close = close.dropna(axis=1, how="all")

            # Separate long-history weekly batch for the all-time-high research field.
            # This is one request for the whole portfolio, not one request per stock.
            ath_highs = {}
            try:
                ath_start = (datetime.now(timezone.utc) - timedelta(days=365 * 20)).date().isoformat()
                ath_params = {
                    "symbols": ",".join(data_symbols),
                    "timeframe": "1Week",
                    "start": ath_start,
                    "limit": 10000,
                    "adjustment": "all",
                    "feed": "iex",
                    "sort": "asc",
                }
                ath_resp = requests.get(
                    data_url, params=ath_params,
                    headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                    timeout=15,
                )
                ath_resp.raise_for_status()
                ath_body = ath_resp.json()
                for sym, rows in (ath_body.get("bars") or {}).items():
                    vals = []
                    for row in rows or []:
                        try:
                            hv = float(row.get("h"))
                            if math.isfinite(hv):
                                vals.append(hv)
                        except (TypeError, ValueError):
                            pass
                    if vals:
                        ath_highs[sym] = max(vals)
            except Exception as ath_ex:
                logger.warning("Portfolio all-time-high batch unavailable: %s", ath_ex)
            rets = close.pct_change(fill_method=None).dropna(how="all")
            for sym in symbols:
                if sym in close.columns:
                    s = close[sym].dropna()
                    rr = rets[sym].dropna() if sym in rets.columns else None
                    if len(s) >= 25:
                        ma20 = float(s.tail(20).mean())
                        ma50 = float(s.tail(50).mean()) if len(s) >= 50 else float(s.mean())
                        last = float(s.iloc[-1])
                        ret20 = (last / float(s.iloc[-21]) - 1) * 100 if len(s) >= 21 else 0
                        ret60 = (last / float(s.iloc[-61]) - 1) * 100 if len(s) >= 61 else 0
                        vol = float(rr.tail(20).std() * math.sqrt(252) * 100) if rr is not None and len(rr) >= 10 else 0
                        peak = float(s.tail(63).max())
                        dd = (last / peak - 1) * 100 if peak else 0
                        score = 0
                        flags = []
                        if last > ma20 * 1.08: score += 2; flags.append("price >8% above 20D mean")
                        if ret20 > 12: score += 2; flags.append("strong 20D run-up")
                        if last < ma20: score += 2; flags.append("below 20D mean")
                        if ma20 < ma50: score += 2; flags.append("20D mean below 50D")
                        if dd < -7: score += 1; flags.append("drawdown from 3M high")
                        if vol > 55: score += 1; flags.append("high realized volatility")
                        highs = pd.Series(high_data.get(sym, {}), dtype="float64").sort_index()
                        high_52w = float(highs.tail(252).max()) if len(highs) else float(s.tail(252).max())
                        all_time_high = ath_highs.get(sym)
                        if all_time_high is None:
                            # Fall back to the longest daily history already loaded.
                            all_time_high = float(highs.max()) if len(highs) else float(s.max())
                        market[sym] = {"last":last,"ma20":ma20,"ma50":ma50,"ret20_pct":ret20,
                                       "high_52w":high_52w,"all_time_high":float(all_time_high),
                                       "ret60_pct":ret60,"volatility_pct":vol,"drawdown_from_3m_high_pct":dd,
                                       "pullback_watch_score":min(score,10),"flags":flags}
            # Use daily adjusted closes for consistent return/correlation windows.
            # 21/63/126/252 observations approximate 1/3/6/12 trading months.
            def _period_return(series, bars):
                series = series.dropna()
                if len(series) <= bars:
                    return None
                base = _safe_float(series.iloc[-(bars + 1)], 0)
                last_px = _safe_float(series.iloc[-1], 0)
                return round((last_px / base - 1) * 100, 2) if base > 0 else None

            def _risk_metrics(sym):
                if sym not in close.columns:
                    return {"ret_1m_pct":None,"ret_3m_pct":None,"ret_6m_pct":None,"ret_12m_pct":None,
                            "volatility_3m_pct":None,"max_drawdown_12m_pct":None,"observations":0,
                            "coverage_12m_pct":0}
                series = close[sym].dropna()
                rr = rets[sym].dropna() if sym in rets.columns else pd.Series(dtype="float64")
                peak = series.tail(252).cummax()
                dd = (series.tail(252) / peak - 1) * 100 if len(series) else pd.Series(dtype="float64")
                return {
                    "ret_1m_pct": _period_return(series, 21),
                    "ret_3m_pct": _period_return(series, 63),
                    "ret_6m_pct": _period_return(series, 126),
                    "ret_12m_pct": _period_return(series, 252),
                    "volatility_3m_pct": round(float(rr.tail(63).std() * math.sqrt(252) * 100), 2) if len(rr) >= 20 else None,
                    "max_drawdown_12m_pct": round(float(dd.min()), 2) if len(series) >= 30 else None,
                    "observations": int(len(series)),
                    "coverage_12m_pct": round(min(100.0, len(series) / 252 * 100), 1),
                }

            risk_by_symbol = {sym: _risk_metrics(sym) for sym in symbols}
            corr_symbols = [s for s in symbols if s in rets.columns and rets[s].count() >= 30]
            pairs = []
            for i, a in enumerate(corr_symbols):
                for b in corr_symbols[i+1:]:
                    aligned = rets[[a,b]].dropna()
                    corr_1m = _safe_float(aligned.tail(21)[a].corr(aligned.tail(21)[b]), float("nan")) if len(aligned.tail(21)) >= 15 else float("nan")
                    corr_6m = _safe_float(aligned.tail(126)[a].corr(aligned.tail(126)[b]), float("nan")) if len(aligned.tail(126)) >= 60 else float("nan")
                    # Keep pairs that are strongly correlated on either horizon, so
                    # recent relationship changes are visible instead of hidden.
                    if (math.isfinite(corr_1m) and corr_1m >= 0.75) or (math.isfinite(corr_6m) and corr_6m >= 0.75):
                        ma, mb = risk_by_symbol[a], risk_by_symbol[b]
                        pairs.append({
                            "a":a,"b":b,
                            "corr_1m":round(corr_1m,2) if math.isfinite(corr_1m) else None,
                            "corr_6m":round(corr_6m,2) if math.isfinite(corr_6m) else None,
                            "a_metrics":ma,"b_metrics":mb,
                            "return_spread_pct":{
                                "1m":round(ma["ret_1m_pct"]-mb["ret_1m_pct"],2) if ma["ret_1m_pct"] is not None and mb["ret_1m_pct"] is not None else None,
                                "3m":round(ma["ret_3m_pct"]-mb["ret_3m_pct"],2) if ma["ret_3m_pct"] is not None and mb["ret_3m_pct"] is not None else None,
                                "6m":round(ma["ret_6m_pct"]-mb["ret_6m_pct"],2) if ma["ret_6m_pct"] is not None and mb["ret_6m_pct"] is not None else None,
                                "12m":round(ma["ret_12m_pct"]-mb["ret_12m_pct"],2) if ma["ret_12m_pct"] is not None and mb["ret_12m_pct"] is not None else None,
                            }
                        })
            pairs.sort(key=lambda x:max(x.get("corr_1m") or -1, x.get("corr_6m") or -1), reverse=True)
            spy_beta = {}
            if "SPY" in rets.columns and rets["SPY"].count() >= 30:
                sr = rets["SPY"].dropna().tail(90)
                var = _safe_float(sr.var())
                if var > 0:
                    for sym in symbols:
                        if sym in rets.columns:
                            joined = rets[[sym]].join(sr.rename("SPY"), how="inner").dropna()
                            if len(joined) >= 25:
                                beta = _safe_float(joined.iloc[:,0].cov(joined["SPY"]) / var)
                                spy_beta[sym] = beta
            total_beta_dollars = sum(abs(p["market_value"]) * spy_beta.get(p["symbol"], 1.0) for p in positions)
            beta_shock = -0.20 * total_beta_dollars
            payload = {
                "as_of": datetime.now(timezone.utc).isoformat(), "account":{"equity":equity,"cash":cash,
                "buying_power":_safe_float(account.get("buying_power")),"portfolio_value":_safe_float(account.get("portfolio_value") or equity)},
                "positions":positions,"theme_exposure":theme_exposure,"summary":{"position_count":len(positions),"market_value":total_mv,
                "invested_pct":invested,"cash_pct":(cash/equity*100 if equity else 0),
                "unrealized_pl":sum(p["unrealized_pl"] for p in positions),
                "top5_weight_pct":sum(p["weight_pct"] for p in positions[:5]),
                "simple_minus20_pct_dollars":-0.20*total_mv,"beta_minus20_estimate":beta_shock,
                "beta_coverage_count":len(spy_beta)},
                "technical":market,"risk_metrics":risk_by_symbol,"high_correlations":pairs[:30],"warnings":warnings,
                "methodology":"Scenario estimates are first-order approximations, not forecasts. Beta defaults to 1 where history is unavailable. Leveraged/inverse ETFs, options, gaps, liquidity and changing correlations can cause materially different outcomes."
            }
        except Exception as ex:
            logger.warning("Portfolio market-data analytics unavailable: %s", ex)
            warnings.append("Historical market data unavailable; showing live positions and basic stress data only.")
            payload = {
                "as_of": datetime.now(timezone.utc).isoformat(),"account":{"equity":equity,"cash":cash,
                "buying_power":_safe_float(account.get("buying_power")),"portfolio_value":_safe_float(account.get("portfolio_value") or equity)},
                "positions":positions,"theme_exposure":theme_exposure,
                "summary":{"position_count":len(positions),"market_value":total_mv,
                "invested_pct":invested,"cash_pct":(cash/equity*100 if equity else 0),
                "unrealized_pl":sum(p["unrealized_pl"] for p in positions),"top5_weight_pct":sum(p["weight_pct"] for p in positions[:5]),
                "simple_minus20_pct_dollars":-0.20*total_mv,"beta_minus20_estimate":-0.20*total_mv,"beta_coverage_count":0},
                "technical":{},"risk_metrics":{},"high_correlations":[],"warnings":warnings,
                "methodology":"Scenario estimates are first-order approximations, not forecasts."
            }
        _CACHE.update({"at":now,"payload":payload})
        return jsonify(payload)
    except Exception as ex:
        logger.exception("Portfolio Manager data error")
        # Safe diagnostic text: never include request headers or credential values.
        return jsonify({"error":"Portfolio data could not be loaded.","detail":str(ex)[:500],
                        "configured":{"alpaca_key_present":bool(_headers().get("APCA-API-KEY-ID")),
                                      "alpaca_secret_present":bool(_headers().get("APCA-API-SECRET-KEY")),
                                      "api_base":(Config.ALPACA_BASE_URL or "").rstrip("/")}}), 503

@portfolio_bp.route("/api/portfolio-manager/ai", methods=["POST"])
def portfolio_ai():
    err = _auth_json()
    if err: return err
    body = request.get_json(silent=True) or {}
    snapshot = body.get("snapshot") or {}
    try:
        # Reuse the AI provider already configured for MirrorFish (Groq/OpenRouter/HuggingFace).
        from mirrorfish.engine import chat, get_provider_status
        status = get_provider_status()
        configured = any(v.get("configured") for v in status.values())
        if not configured:
            return jsonify({"available":False,"message":"No existing MirrorFish AI provider is configured. The quantitative dashboard works without AI."}), 200
        prompt = ("Review this portfolio risk snapshot as a cautious hedge-fund risk analyst. "
                  "Use only the supplied figures. Distinguish observed facts from hypotheses; "
                  "do not claim to predict tops or guarantee a pullback. Discuss concentration, "
                  "correlation, 20% market stress assumptions, and which profit-taking candidates "
                  "deserve a manual review. Do not recommend automatic orders. Keep the answer concise. "
                  "Snapshot JSON: " + __import__("json").dumps(snapshot, separators=(",",":"))[:16000])
        # Prefer the same gpt-oss-20b model used by the hosted Hugging Face route
        # when its token is configured; otherwise retain the existing provider fallback.
        preferred = "huggingface" if status.get("huggingface", {}).get("configured") else None
        answer = chat(prompt, {"module":"Portfolio Manager","as_of":snapshot.get("as_of")}, provider_name=preferred)
        return jsonify({"available":True,"analysis":str(answer)[:12000]})
    except Exception as ex:
        logger.warning("Portfolio AI unavailable: %s",ex)
        return jsonify({"available":False,"message":"Existing MirrorFish AI endpoint unavailable; use the quantitative dashboard below.","detail":str(ex)}),200
