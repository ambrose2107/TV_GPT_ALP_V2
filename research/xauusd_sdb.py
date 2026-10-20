"""Isolated ICT-style Standard Deviation (SDB) research engine for GLD.

This is an explicitly documented research approximation, not a claim to reproduce
every TradingView/ICT deviation tool. It uses real Alpaca GLD 5m stock bars,
never synthetic data. All higher-timeframe context is derived from completed
5m bars. Signals are entered at the next bar open in the backtest.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def load_gold_data(period: str = "60d") -> pd.DataFrame:
    """Fetch real Alpaca GLD 5m bars. Requires Alpaca credentials; never falls back."""
    import os
    from datetime import datetime, timedelta, timezone
    import requests

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials are missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "(or ALPACA_API_KEY and ALPACA_API_SECRET_KEY) in the Render environment."
        )
    try:
        days = int(str(period).lower().removesuffix("d"))
    except (TypeError, ValueError):
        days = 60
    days = min(max(days, 1), 60)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = "https://data.alpaca.markets/v2/stocks/GLD/bars"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": "5Min",
        "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "end": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "limit": 10000,
        "adjustment": "raw",
        "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
        "sort": "asc",
    }
    records = []
    page_token = None
    try:
        while True:
            if page_token:
                params["page_token"] = page_token
            response = requests.get(url, headers=headers, params=params, timeout=25)
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "Alpaca rejected the GLD data request. Check Render API key/secret and "
                    "whether the account has access to the selected market-data feed."
                )
            response.raise_for_status()
            payload = response.json()
            bars_payload = payload.get("bars", [])
            if isinstance(bars_payload, dict):
                records.extend(bars_payload.get("GLD", []) or [])
            elif isinstance(bars_payload, list):
                records.extend(bars_payload)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    except RuntimeError:
        raise
    except requests.RequestException as exc:
        raise RuntimeError(f"Alpaca GLD data request failed: {exc}") from exc
    if not records:
        raise RuntimeError(
            "Alpaca returned no GLD 5m bars for the requested window. Check market-data "
            "entitlements, feed setting, and Render credentials. No synthetic fallback is permitted."
        )

    raw = pd.DataFrame(records)
    required_api = {"t", "o", "h", "l", "c"}
    missing_api = required_api.difference(raw.columns)
    if missing_api:
        raise RuntimeError("Alpaca GLD response is missing fields: " + ", ".join(sorted(missing_api)))
    raw = raw.rename(columns={"t": "Time", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    raw["Time"] = pd.to_datetime(raw["Time"], utc=True, errors="coerce")
    raw = raw.set_index("Time")
    if "Volume" not in raw.columns:
        raw["Volume"] = 0.0
    out = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(out) < 500:
        raise RuntimeError(f"Alpaca returned only {len(out)} real GLD 5m bars; at least 500 are required.")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise RuntimeError("Alpaca GLD data index is not datetime.")
    return out


def _resample(ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Yahoo intraday timestamps mark bar opens. Group left-closed bars, then
    # shift the aggregate timestamp to the period close so MTF values are not
    # visible to 5m logic before the constituent bars have completed.
    out = ohlc.resample(rule, label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])
    out.index = out.index + pd.Timedelta(rule)
    return out


def build_sdb_signals(data: pd.DataFrame, sweep_lookback: int = 12,
                      mss_lookback: int = 3, mss_window: int = 6,
                      h1_ema_len: int = 20, atr_len: int = 14,
                      stop_buffer_atr: float = 0.15) -> pd.DataFrame:
    """Return 5m rows with non-lookahead signals, structural stops and SD targets."""
    if data is None or data.empty:
        raise ValueError("No gold bars supplied.")
    d = data.copy().sort_index()
    d = d[~d.index.duplicated(keep="last")]
    for c in ("Open", "High", "Low", "Close"):
        if c not in d.columns:
            raise ValueError(f"Missing required price column: {c}")
    m15 = _resample(d, "15min")
    h1 = _resample(d, "1h")
    if len(m15) < sweep_lookback + 5 or len(h1) < h1_ema_len + 2:
        raise ValueError("Insufficient 15m/1H history for the configured SDB indicators.")

    h1["ema"] = h1["Close"].ewm(span=h1_ema_len, adjust=False, min_periods=h1_ema_len).mean()
    h1["bias"] = np.where((h1["Close"] > h1["ema"]) & (h1["ema"] > h1["ema"].shift(1)), 1,
                   np.where((h1["Close"] < h1["ema"]) & (h1["ema"] < h1["ema"].shift(1)), -1, 0))
    # The 1H value is only made available at its resampled close timestamp.
    bias_frame = h1[["bias"]].reset_index()
    bias_frame.columns = ["time", "bias"]
    five = d.reset_index()
    five.columns = ["time"] + list(d.columns)
    five = pd.merge_asof(five.sort_values("time"), bias_frame.sort_values("time"),
                         on="time", direction="backward")
    five = five.set_index("time")
    five["bias"] = five["bias"].fillna(0).astype(int)

    # ATR from 5m bars, used only for stop buffer and diagnostics.
    prev_close = five["Close"].shift(1)
    tr = pd.concat([(five["High"] - five["Low"]),
                    (five["High"] - prev_close).abs(),
                    (five["Low"] - prev_close).abs()], axis=1).max(axis=1)
    five["atr"] = tr.rolling(atr_len, min_periods=atr_len).mean()

    # 15m liquidity sweep: current wick crosses a prior range extreme and closes back inside.
    m15["prior_low"] = m15["Low"].rolling(sweep_lookback, min_periods=sweep_lookback).min().shift(1)
    m15["prior_high"] = m15["High"].rolling(sweep_lookback, min_periods=sweep_lookback).max().shift(1)
    m15["bull_sweep"] = (m15["Low"] < m15["prior_low"]) & (m15["Close"] > m15["prior_low"])
    m15["bear_sweep"] = (m15["High"] > m15["prior_high"]) & (m15["Close"] < m15["prior_high"])

    # A completed 15m sweep can only trigger 5m MSS on/after its close timestamp.
    prior5_high = five["High"].rolling(mss_lookback, min_periods=mss_lookback).max().shift(1)
    prior5_low = five["Low"].rolling(mss_lookback, min_periods=mss_lookback).min().shift(1)
    five["signal"] = 0
    five["stop"] = np.nan
    five["sd1"] = np.nan
    five["sd2"] = np.nan
    five["sd25"] = np.nan
    five["sd4"] = np.nan
    five["setup"] = ""
    five["sweep_time"] = ""
    five["leg_range"] = np.nan
    five["mss_level"] = np.nan

    idx = five.index
    # Keep only the latest active sweep per direction; all decisions use bars already closed.
    events = []
    for t, row in m15.iterrows():
        if bool(row["bull_sweep"]) and pd.notna(row["prior_low"]):
            events.append({"time": t, "side": 1, "extreme": float(row["Low"]),
                           "level": float(row["prior_low"]), "range": float(row["High"] - row["Low"])})
        if bool(row["bear_sweep"]) and pd.notna(row["prior_high"]):
            events.append({"time": t, "side": -1, "extreme": float(row["High"]),
                           "level": float(row["prior_high"]), "range": float(row["High"] - row["Low"])})
    events.sort(key=lambda x: x["time"])
    used = set()
    for ev in events:
        start = idx.searchsorted(ev["time"], side="left")
        if start >= len(idx):
            continue
        end = min(len(idx), start + max(1, int(mss_window)))
        for j in range(start, end):
            if j in used:
                continue
            row = five.iloc[j]
            if int(row["bias"]) != ev["side"]:
                continue
            if ev["side"] == 1:
                mss = pd.notna(prior5_high.iloc[j]) and row["Close"] > prior5_high.iloc[j]
            else:
                mss = pd.notna(prior5_low.iloc[j]) and row["Close"] < prior5_low.iloc[j]
            if not mss or not np.isfinite(row["atr"]):
                continue
            entry_ref = float(row["Close"])
            # Explicit research convention: project positive range multiples in trade direction.
            # The UI labels them as ICT-style negative SD levels per the user's description;
            # they are not statistical sigma estimates from a return distribution.
            leg = max(float(ev["range"]), abs(entry_ref - ev["extreme"]), float(row["atr"]))
            buffer = float(stop_buffer_atr) * float(row["atr"])
            stop = ev["extreme"] - buffer if ev["side"] == 1 else ev["extreme"] + buffer
            risk = abs(entry_ref - stop)
            if not np.isfinite(risk) or risk <= 0:
                continue
            five.iloc[j, five.columns.get_loc("signal")] = ev["side"]
            five.iloc[j, five.columns.get_loc("stop")] = stop
            five.iloc[j, five.columns.get_loc("sd1")] = entry_ref + ev["side"] * leg
            five.iloc[j, five.columns.get_loc("sd2")] = entry_ref + ev["side"] * 2.0 * leg
            five.iloc[j, five.columns.get_loc("sd25")] = entry_ref + ev["side"] * 2.5 * leg
            five.iloc[j, five.columns.get_loc("sd4")] = entry_ref + ev["side"] * 4.0 * leg
            five.iloc[j, five.columns.get_loc("setup")] = "bullish sweep + MSS" if ev["side"] == 1 else "bearish sweep + MSS"
            five.iloc[j, five.columns.get_loc("sweep_time")] = str(ev["time"])
            five.iloc[j, five.columns.get_loc("leg_range")] = leg
            five.iloc[j, five.columns.get_loc("mss_level")] = float(prior5_high.iloc[j] if ev["side"] == 1 else prior5_low.iloc[j])
            used.add(j)
            break
    return five


def run_backtest(data: pd.DataFrame, initial_equity: float = 10000.0,
                 risk_pct: float = 0.5, spread_points: float = 0.30,
                 slippage_points: float = 0.05, commission_pct: float = 0.0005,
                 **signal_params) -> dict:
    """Simple conservative next-open backtest with 4 equal partial target exits."""
    if not (0 < risk_pct <= 2.0):
        raise ValueError("risk_pct must be > 0 and <= 2.")
    d = build_sdb_signals(data, **signal_params)
    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    trades = []
    active = None
    curve = []
    for i in range(1, len(d)):
        prev = d.iloc[i - 1]
        bar = d.iloc[i]
        ts = d.index[i]
        if active is not None:
            side = active["side"]
            stop_hit = bar["Low"] <= active["stop"] if side == 1 else bar["High"] >= active["stop"]
            targets = active["targets"]
            next_target = active["next_target"]
            target_hit = next_target < len(targets) and (
                bar["High"] >= targets[next_target] if side == 1 else bar["Low"] <= targets[next_target])
            # If both stop and target are inside the same OHLC bar, assume stop first.
            if stop_hit:
                px = active["stop"] - side * (spread_points / 2 + slippage_points)
                frac = active["remaining"]
                pnl = (px - active["entry"]) * active["units"] * side * frac
                pnl -= abs(px * active["units"] * frac) * commission_pct
                equity += pnl
                active["pnl"] += pnl
                active["remaining"] = 0.0
                active["exit_time"] = str(ts)
                active["exit_reason"] = "stop"
            elif target_hit:
                target = targets[next_target]
                px = target - side * (spread_points / 2 + slippage_points)
                frac = min(0.25, active["remaining"])
                pnl = (px - active["entry"]) * active["units"] * side * frac
                pnl -= abs(px * active["units"] * frac) * commission_pct
                equity += pnl
                active["pnl"] += pnl
                active["remaining"] -= frac
                active["next_target"] += 1
                if active["next_target"] == 1:
                    active["stop"] = active["entry"]
                if active["remaining"] < 1e-8:
                    active["remaining"] = 0.0
                    active["exit_time"] = str(ts)
                    active["exit_reason"] = "all_targets"
            if active["remaining"] <= 0:
                active["equity_after"] = equity
                trades.append(active.copy())
                active = None
        if active is None and int(prev.get("signal", 0)) != 0:
            side = int(prev["signal"])
            entry = float(bar["Open"]) + side * (spread_points / 2 + slippage_points)
            stop = float(prev["stop"])
            risk_dist = abs(entry - stop)
            if not np.isfinite(risk_dist) or risk_dist <= 0:
                curve.append({"time": str(ts), "equity": equity})
                continue
            risk_money = equity * risk_pct / 100.0
            units = risk_money / risk_dist
            targets = [float(prev[k]) for k in ("sd1", "sd2", "sd25", "sd4")]
            targets = [x for x in targets if (x > entry if side == 1 else x < entry)]
            if not targets:
                curve.append({"time": str(ts), "equity": equity})
                continue
            active = {
                "side": side, "entry_time": str(ts), "entry": entry, "stop": stop,
                "targets": targets, "next_target": 0, "units": units, "remaining": 1.0,
                "pnl": 0.0, "setup": str(prev.get("setup", "")),
                "sweep_time": str(prev.get("sweep_time", "")),
                "leg_range": float(prev.get("leg_range", np.nan)),
                "initial_risk": risk_money, "exit_time": "", "exit_reason": "open",
            }
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
        curve.append({"time": str(ts), "equity": equity})
    if active is not None:
        last = d.iloc[-1]
        side = active["side"]
        px = float(last["Close"])
        frac = active["remaining"]
        pnl = (px - active["entry"]) * active["units"] * side * frac
        pnl -= abs(px * active["units"] * frac) * commission_pct
        equity += pnl
        active["pnl"] += pnl
        active["exit_time"] = str(d.index[-1])
        active["exit_reason"] = "end_of_data"
        active["remaining"] = 0.0
        active["equity_after"] = equity
        trades.append(active.copy())

    wins = sum(1 for t in trades if t["pnl"] > 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    metrics = {
        "bars": int(len(d)), "trades": int(len(trades)),
        "win_rate_pct": round(100 * wins / len(trades), 2) if trades else 0.0,
        "net_pnl": round(sum(t["pnl"] for t in trades), 2),
        "return_pct": round((equity / initial_equity - 1) * 100, 2) if initial_equity else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (None if gross_profit == 0 else "inf"),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_equity": round(equity, 2),
        "data_start": str(d.index.min()), "data_end": str(d.index.max()),
        "data_source": "Alpaca GLD 5m ETF bars (gold ETF proxy; not spot XAUUSD)",
    }
    serial_trades = []
    for t in trades[-500:]:
        x = dict(t)
        x["targets"] = x["targets"]
        x["pnl"] = round(float(x["pnl"]), 2)
        x["equity_after"] = round(float(x.get("equity_after", equity)), 2)
        serial_trades.append(x)
    return {"metrics": metrics, "trades": serial_trades,
            "signals": d[d["signal"] != 0].tail(500).reset_index().rename(columns={"index": "time"}).replace({np.nan: None}).to_dict("records"),
            "equity_curve": curve[-2000:]}
