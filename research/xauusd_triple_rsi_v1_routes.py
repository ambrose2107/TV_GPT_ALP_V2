"""Routes for the daily Triple-RSI reference strategy."""
import math
from flask import Blueprint, jsonify, request, render_template, session

from core.market_data import alpaca_get_bars, get_bars
from research.xauusd_triple_rsi_v1 import TripleRSIConfig, backtest

xauusd_triple_rsi_v1_bp = Blueprint("xauusd_triple_rsi_v1", __name__)


def _safe(v):
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if hasattr(v, "item"):
        try:
            return _safe(v.item())
        except Exception:
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


@xauusd_triple_rsi_v1_bp.route("/xauusd-triple-rsi-v1")
def xauusd_triple_rsi_v1_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_triple_rsi_v1.html")


@xauusd_triple_rsi_v1_bp.route("/api/xauusd-triple-rsi-v1/run", methods=["POST"])
def xauusd_triple_rsi_v1_run():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "SPY")).strip().upper()
    n_bars = max(250, min(5000, int(body.get("n_bars", 1000))))
    raw_cfg = body.get("config") or {}
    allowed = {k: raw_cfg[k] for k in raw_cfg if k in TripleRSIConfig.__dataclass_fields__}
    cfg = TripleRSIConfig(**allowed)

    raw = alpaca_get_bars(symbol, "1Day", limit=n_bars)
    source = "Alpaca"
    if not raw:
        raw = get_bars(symbol, "1y")
        source = "Analyzer Pro/Yahoo fallback"
    if not raw:
        return jsonify({"error": f"No daily data available for {symbol}"}), 404

    import pandas as pd
    df = pd.DataFrame(raw)
    if {"t","o","h","l","c"}.issubset(df.columns):
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = df.set_index("t").sort_index().rename(
            columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}
        )
    result = backtest(df, cfg)

    return jsonify(_safe({
        "strategy": "Triple RSI Mean Reversion V1",
        "symbol": symbol,
        "data_source": source,
        "bars": int(len(df)),
        "data_start": df.index[0].isoformat() if len(df) else None,
        "data_end": df.index[-1].isoformat() if len(df) else None,
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
        "trades": result["trades"],
        "signals": result["signals"].tail(300).to_dict("records"),
        "daily_bars": int(len(result["data"])),
        "rules": {
            "rsi": "RSI(5) < 30",
            "third_decline": "RSI(5) lower than the prior two sessions",
            "three_days_ago": "RSI(5) was < 60 three sessions ago",
            "trend": "Close > SMA200",
            "entry": "buy at the completed daily close",
            "exit": "sell at the completed daily close when RSI(5) crosses above 50",
        },
    }))


__all__ = ["xauusd_triple_rsi_v1_bp"]
