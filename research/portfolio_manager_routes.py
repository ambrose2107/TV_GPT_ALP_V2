"""Portfolio Manager: lightweight portfolio risk and profit-taking research."""
from flask import Blueprint, jsonify, render_template, request, session
from core.config import Config
from core.logger import get_logger
import os, math, time
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
        import requests
        base = (Config.ALPACA_BASE_URL or "").rstrip("/")
        headers = _headers()
        if not headers.get("APCA-API-KEY-ID") or not headers.get("APCA-API-SECRET-KEY"):
            raise RuntimeError("Alpaca credentials are missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY (or APCA_API_KEY_ID and APCA_API_SECRET_KEY) in Render Environment, then redeploy.")
        if not base:
            raise RuntimeError("Alpaca API base URL is empty. Check ALPACA_MODE.")
        try:
            ar = requests.get(base + "/v2/account", headers=headers, timeout=10)
        except requests.RequestException as ex:
            raise RuntimeError("Could not connect to Alpaca account API. Check Render outbound connectivity and ALPACA_MODE. Details: " + str(ex)[:240])
        if ar.status_code in (401, 403):
            raise RuntimeError("Alpaca rejected the credentials for " + base + " (HTTP " + str(ar.status_code) + "). Check that the key/secret pair matches this environment: paper keys for ALPACA_MODE=paper, live keys for ALPACA_MODE=live. No credentials were displayed.")
        if not ar.ok:
            raise RuntimeError("Alpaca account API returned HTTP " + str(ar.status_code) + ": " + ar.text[:240])
        try:
            pr = requests.get(base + "/v2/positions", headers=headers, timeout=10)
        except requests.RequestException as ex:
            raise RuntimeError("Connected to Alpaca account, but positions request failed: " + str(ex)[:240])
        if pr.status_code in (401, 403):
            raise RuntimeError("Alpaca rejected the positions request (HTTP " + str(pr.status_code) + "). Verify the API key/secret and account permissions.")
        if not pr.ok:
            raise RuntimeError("Alpaca positions API returned HTTP " + str(pr.status_code) + ": " + pr.text[:240])
        account, raw = ar.json(), pr.json()
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
        # One bounded daily-bar request; avoid large per-ticker calls on Render free tier.
        symbols = list(dict.fromkeys([p["symbol"] for p in positions if p["symbol"]]))[:35]
        market = {}
        warnings = []
        try:
            import yfinance as yf
            tickers = list(dict.fromkeys(symbols + ["SPY", "QQQ", "IWM", "TLT", "GLD"]))
            hist = yf.download(tickers=tickers, period="6mo", interval="1d",
                               auto_adjust=True, progress=False, group_by="ticker",
                               threads=False, timeout=12)
            if hist is not None and not hist.empty:
                import pandas as pd
                if isinstance(hist.columns, pd.MultiIndex):
                    if "Close" in hist.columns.get_level_values(-1):
                        close = hist.xs("Close", axis=1, level=-1)
                    elif "Close" in hist.columns.get_level_values(0):
                        close = hist.xs("Close", axis=1, level=0)
                    else:
                        close = None
                else:
                    close = hist[["Close"]] if "Close" in hist.columns else None
                if close is None:
                    raise ValueError("Could not identify adjusted close prices in market-data response")
                if isinstance(close, pd.Series): close = close.to_frame()
                close = close.dropna(axis=1, how="all")
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
                            # Pullback watch is evidence-weighted, not a price-top prediction.
                            score = 0
                            flags = []
                            if last > ma20 * 1.08: score += 2; flags.append("price >8% above 20D mean")
                            if ret20 > 12: score += 2; flags.append("strong 20D run-up")
                            if last < ma20: score += 2; flags.append("below 20D mean")
                            if ma20 < ma50: score += 2; flags.append("20D mean below 50D")
                            if dd < -7: score += 1; flags.append("drawdown from 3M high")
                            if vol > 55: score += 1; flags.append("high realized volatility")
                            market[sym] = {"last":last,"ma20":ma20,"ma50":ma50,"ret20_pct":ret20,
                                           "ret60_pct":ret60,"volatility_pct":vol,"drawdown_from_3m_high_pct":dd,
                                           "pullback_watch_score":min(score,10),"flags":flags}
                # Correlations from daily returns (only symbols with adequate data).
                corr_symbols = [s for s in symbols if s in rets.columns and rets[s].count() >= 30]
                corr = rets[corr_symbols].tail(90).corr() if len(corr_symbols) >= 2 else None
                pairs = []
                if corr is not None:
                    for i, a in enumerate(corr_symbols):
                        for b in corr_symbols[i+1:]:
                            c = _safe_float(corr.loc[a,b], 0)
                            if c >= 0.75: pairs.append({"a":a,"b":b,"corr":round(c,2)})
                pairs.sort(key=lambda x:x["corr"], reverse=True)
                # Benchmark beta and a transparent first-order 20% SPY shock.
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
                # Correlation is descriptive; sector labels are deliberately not guessed from ticker.
                payload = {
                    "as_of": datetime.now(timezone.utc).isoformat(), "account":{"equity":equity,"cash":cash,
                    "buying_power":_safe_float(account.get("buying_power")),"portfolio_value":_safe_float(account.get("portfolio_value") or equity)},
                    "positions":positions,"theme_exposure":theme_exposure,"summary":{"position_count":len(positions),"market_value":total_mv,
                    "invested_pct":invested,"cash_pct":(cash/equity*100 if equity else 0),
                    "unrealized_pl":sum(p["unrealized_pl"] for p in positions),
                    "top5_weight_pct":sum(p["weight_pct"] for p in positions[:5]),
                    "simple_minus20_pct_dollars":-0.20*total_mv,"beta_minus20_estimate":beta_shock,
                    "beta_coverage_count":len(spy_beta)},
                    "technical":market,"high_correlations":pairs[:30],"warnings":warnings,
                    "methodology":"Scenario estimates are first-order approximations, not forecasts. Beta defaults to 1 where history is unavailable. Leveraged/inverse ETFs, options, gaps, liquidity and changing correlations can cause materially different outcomes."
                }
        except Exception as ex:
            logger.warning("Portfolio history analytics unavailable: %s", ex)
            warnings.append("Historical price data unavailable; showing live position and basic stress data only.")
            payload = {
                "as_of": datetime.now(timezone.utc).isoformat(),"account":{"equity":equity,"cash":cash,
                "buying_power":_safe_float(account.get("buying_power")),"portfolio_value":_safe_float(account.get("portfolio_value") or equity)},
                "positions":positions,"summary":{"position_count":len(positions),"market_value":total_mv,
                "invested_pct":invested,"cash_pct":(cash/equity*100 if equity else 0),
                "unrealized_pl":sum(p["unrealized_pl"] for p in positions),"top5_weight_pct":sum(p["weight_pct"] for p in positions[:5]),
                "simple_minus20_pct_dollars":-0.20*total_mv,"beta_minus20_estimate":-0.20*total_mv,"beta_coverage_count":0},
                "technical":{},"high_correlations":[],"warnings":warnings,
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
        answer = chat(prompt, {"module":"Portfolio Manager","as_of":snapshot.get("as_of")})
        return jsonify({"available":True,"analysis":str(answer)[:12000]})
    except Exception as ex:
        logger.warning("Portfolio AI unavailable: %s",ex)
        return jsonify({"available":False,"message":"Existing MirrorFish AI endpoint unavailable; use the quantitative dashboard below.","detail":str(ex)}),200
