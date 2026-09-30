"""Authenticated API for the focused V2 research candidates.

V2 contains only the newer research engines. The older seven-strategy lab and
its legacy runner remain isolated in the V1 page/backend paths.
"""
import math
import pandas as pd
import time
import io
import csv
import json
import uuid
import itertools
import zipfile
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, session, send_file, render_template, redirect, url_for

from core.market_data import alpaca_get_bars, get_bars
from research.xauusd_pullback_v2 import PullbackV2Config, backtest as pullback_v2_backtest
from research.xauusd_ema_retest_v2 import EMARetestV2Config, backtest as ema_v2_backtest
from research.xauusd_trend_target_ribbon_v2 import TrendTargetRibbonConfig, backtest as trend_ribbon_backtest
from research.xauusd_daily_research_v2 import (
    WilliamsRConfig, CCIConfig, MultiHorizonRSIConfig,
    backtest_williams_r, backtest_cci, backtest_multi_rsi,
)
from research.xauusd_confluence_v4 import load_data as v4_load_data
from core.order_sync import sync_alpaca_orders
from core.database import get_all_trades, get_all_closed_positions, get_closed_summary

bp = Blueprint("xauusd_research_v2", __name__)
_DATA_CACHE = {}
_CACHE_TTL = 300
_LAST_RUN = {"status": "never", "updated": None, "summary": {}}
_RUN_DATA = {}


def _excel_safe_value(v):
    """Excel cannot store timezone-aware datetime values; normalize them to UTC-naive."""
    if isinstance(v, pd.Timestamp):
        if v.tzinfo is not None:
            return v.tz_convert("UTC").tz_localize(None)
        return v
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            return v.astimezone(timezone.utc).replace(tzinfo=None)
        return v
    return v


def _excel_safe_df(df):
    """Return a copy safe for openpyxl, including object columns containing tz datetimes."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_excel_safe_value)
    return out


def _safe(v):
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
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


def _cfg(cls, raw):
    raw = raw or {}
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _bars_to_df(raw, symbol, timeframe):
    if not raw:
        raise ValueError(f"No {timeframe} market data returned for {symbol}")

    df = pd.DataFrame(raw)
    rename = {"t": "timestamp", "o": "Open", "h": "High", "l": "Low",
              "c": "Close", "v": "Volume"}
    df = df.rename(columns=rename)

    # Accept both Alpaca's compact keys and any normalized bars returned by a
    # fallback provider.
    required = {"timestamp", "Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Market-data format error for {symbol} {timeframe}: "
            f"missing {sorted(missing)}; columns={list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df = (
        df.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])
          .set_index("timestamp")
          .sort_index()
    )
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _intraday(symbol, n):
    """Load enough 5m history for V2, with a proven paginated fallback."""
    n = max(500, min(30000, int(n)))
    key = ("5m", symbol, n)
    cached = _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1].copy(), cached[2]
    try:
        raw = alpaca_get_bars(symbol, "5Min", limit=n)
        if raw:
            df = _bars_to_df(raw, symbol, "5m")
            if len(df) >= 500:
                out = df.tail(n)
                _DATA_CACHE[key] = (time.time(), out.copy(), "Alpaca")
                return out, "Alpaca"
    except Exception:
        pass

    try:
        df = v4_load_data(use_live=True, n_bars=n, symbol=symbol,
                          data_source="alpaca")["m5"]
        if df is not None and len(df) >= 500:
            out = df.tail(n)
            _DATA_CACHE[key] = (time.time(), out.copy(), "Alpaca/V4 paginated fallback")
            return out, "Alpaca/V4 paginated fallback"
        fallback_error = "fallback returned insufficient bars"
    except Exception as exc:
        fallback_error = str(exc)

    raise ValueError(
        f"V2 could not obtain enough 5m history for {symbol}. "
        f"Need at least 500 bars. {fallback_error}"
    )


def _daily(symbol, n):
    n = max(250, min(5000, int(n)))
    key = ("1d", symbol, n)
    cached = _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1].copy(), cached[2]
    raw = alpaca_get_bars(symbol, "1Day", limit=n)
    source = "Alpaca"
    if not raw:
        raw = get_bars(symbol, "1y")
        source = "Analyzer Pro fallback"
    out = _bars_to_df(raw, symbol, "1Day").tail(n)
    _DATA_CACHE[key] = (time.time(), out.copy(), source)
    return out, source



_STRATEGY_META = {
    "pullback": {"name":"Pullback Breakout V2","subtitle":"Trend + controlled pullback + displacement breakout","description":"EMA trend separation, ATR regime filtering, controlled pullback and displacement breakout with next-bar execution.","default_symbol":"SPY","chart_tf":"15m","kind":"intraday"},
    "ema": {"name":"EMA 20/50 Third Retest V2","subtitle":"20/50 direction + distinct third retest + 1H confirmation","description":"Confirmed 20/50 direction, distinct retest events, rejection quality, EMA separation and completed 1H alignment.","default_symbol":"SPY","chart_tf":"15m","kind":"intraday"},
    "triple": {"name":"Triple RSI — Multi-Horizon","subtitle":"RSI(5) + RSI(14) + RSI(50)","description":"Daily long-only mean reversion using RSI(5) < 45, RSI(14) < 65, RSI(50) < 55, with RSI-based exits.","default_symbol":"SPY","chart_tf":"1d","kind":"daily"},
    "williams": {"name":"Williams %R Mean Reversion","subtitle":"Extreme oversold recovery","description":"Daily long-only mean reversion using Williams %R(2) < -98, price above the 175-day moving average, next-session entry and Williams %R recovery exit.","default_symbol":"SPY","chart_tf":"1d","kind":"daily"},
    "cci": {"name":"CCI Oversold Recovery","subtitle":"CCI(16) extreme oversold recovery","description":"Daily long-only recovery strategy: CCI(16) crosses back above -180, buy next session open, exit when CCI > +150.","default_symbol":"SPY","chart_tf":"1d","kind":"daily"},
    "trend_ribbon": {"name":"Trend Target Ribbon V2","subtitle":"ALMA trend + deviation confirmation + ATR targets","description":"BOSWaves-derived ALMA trend-flip strategy with ATR-normalized slope, deviation confirmation, structure/ATR stop and 1R–4R target diagnostics.","default_symbol":"SPY","chart_tf":"5m","kind":"intraday"},
}

# Controls exposed on each strategy's personal research page. Keep these tied
# to actual dataclass fields so the UI cannot silently send unsupported params.
_STRATEGY_PARAMS = {
    "pullback": [
        ("ema_fast","EMA fast",50,"int",5,200),("ema_slow","EMA slow",200,"int",50,400),
        ("pullback_bars","Pullback bars",3,"int",1,10),("breakout_lookback","Breakout lookback",5,"int",2,20),
        ("atr_stop","ATR stop",1.2,"float",0.25,4.0),("rr","Reward / risk",2.0,"float",0.5,5.0),
        ("cooldown_bars","Cooldown bars",8,"int",0,50),("session_start_utc","Session start UTC",13,"int",0,23),
        ("session_end_utc","Session end UTC",21,"int",1,24),("min_ema_gap_atr","Min EMA gap ATR",0.10,"float",0,2),
        ("breakout_buffer_atr","Breakout buffer ATR",0.15,"float",0,1),("min_body_atr","Min candle body ATR",0.40,"float",0,2),
        ("min_atr_pct","Min ATR %",0.00125,"float",0,0.02),("max_atr_pct","Max ATR %",0.02,"float",0.001,0.10),
    ],
    "ema": [("ema_fast","EMA fast",20,"int",5,100),("ema_slow","EMA slow",50,"int",20,200),("min_gap_atr","Min gap ATR",0.10,"float",0,2),("min_rejection_body_atr","Min rejection body ATR",0.10,"float",0,2),("rr","Reward / risk",2.5,"float",0.5,5),("separation_bars","Separation bars",2,"int",1,10)],
    "trend_ribbon": [("alma_len","ALMA length",34,"int",5,100),("alma_offset","ALMA offset",0.85,"float",0.1,0.99),("alma_sigma","ALMA sigma",6.0,"float",1,15),("dev_len","Deviation length",34,"int",5,100),("dev_mult","Deviation multiplier",0.65,"float",0,3),("slope_len","Slope length",3,"int",1,20),("slope_min","Minimum slope",0.08,"float",0,1),("stop_lookback","Stop lookback",12,"int",2,50),("min_stop_atr","Min stop ATR",0.75,"float",0.1,5),("max_stop_atr","Max stop ATR",3.0,"float",0.5,8),("max_entry_body_atr","Max entry body ATR",1.0,"float",0.25,4),("target_count","Target count",4,"int",2,4),("max_hold_bars","Max hold bars (0=off)",0,"int",0,500),("cooldown_bars","Cooldown bars",0,"int",0,50)],
    "triple": [("rsi_fast","RSI fast",5,"int",2,20),("rsi_mid","RSI mid",14,"int",5,50),("rsi_slow","RSI slow",50,"int",20,100),("entry_fast_max","Fast RSI max",45,"float",1,99),("entry_mid_max","Mid RSI max",65,"float",1,99),("entry_slow_max","Slow RSI max",55,"float",1,99),("exit_fast","Fast RSI exit",90,"float",1,99),("exit_mid","Mid RSI exit",65,"float",1,99)],
    "williams": [("length","Williams length",2,"int",2,20),("entry_level","Entry level",-98,"float",-100,-50),("ma_len","MA length",175,"int",20,500),("exit_level","Exit level",-50,"float",-99,0)],
    "cci": [("length","CCI length",16,"int",5,50),("entry_level","Entry level",-180,"float",-400,0),("exit_level","Exit level",150,"float",0,400)],
};

@bp.route("/xauusd-research-v2/<strategy>", methods=["GET"])
@bp.route("/xauusd-pullback-v2", defaults={"strategy":"pullback"}, methods=["GET"])
@bp.route("/xauusd-ema-retest-v2", defaults={"strategy":"ema"}, methods=["GET"])
@bp.route("/xauusd-trend-ribbon-v2", defaults={"strategy":"trend_ribbon"}, methods=["GET"])
@bp.route("/xauusd-triple-rsi-v2", defaults={"strategy":"triple"}, methods=["GET"])
@bp.route("/xauusd-williams-v2", defaults={"strategy":"williams"}, methods=["GET"])
@bp.route("/xauusd-cci-v2", defaults={"strategy":"cci"}, methods=["GET"])
def strategy_detail(strategy):
    if not session.get("logged_in"):
        return redirect(url_for("dashboard.login"))
    key = str(strategy).strip().lower()
    meta = _STRATEGY_META.get(key)
    if not meta:
        return jsonify({"error": f"Unknown V2 strategy: {key}"}), 404
    return render_template("xauusd_strategy_detail.html", strategy_key=key, strategy_meta=meta, strategy_params=_STRATEGY_PARAMS.get(key, []), chart_symbol=meta["default_symbol"], chart_default_tf=meta["chart_tf"], chart_strategy=key)


@bp.route("/api/xauusd-research-v2/status", methods=["GET"])
def status():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_safe(_LAST_RUN))




@bp.route("/api/strategy-lab/optimize", methods=["POST"])
def optimize_strategy_lab_alias():
    """Compatibility alias for the V2 optimizer used by older deployed pages."""
    return optimize()

@bp.route("/api/xauusd-research-v2/optimize", methods=["POST"])
def optimize():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    strategy = str(body.get("strategy", "pullback")).strip().lower()
    symbol = str(body.get("symbol", "SPY")).strip().upper()
    bars = max(1000, min(30000, int(body.get("n_bars", 15600))))
    train_pct = min(0.8, max(0.5, float(body.get("train_pct", 0.7))))
    min_train = max(5, int(body.get("min_train_trades", 10)))
    min_test = max(3, int(body.get("min_test_trades", 5)))
    # Pullback optimization can be memory-heavy on small Render workers.\n    # Split the eight Stage-1 seeds across independent requests when requested.\n    batch_index = max(0, int(body.get("batch_index", 0)))\n    batch_total = max(1, min(8, int(body.get("batch_total", 1))))\n    if batch_index >= batch_total:\n        return jsonify({"error": "Invalid optimizer batch index"}), 400\n    try:
        df, source = _intraday(symbol, bars)
        split = max(500, min(len(df)-100, int(len(df) * train_pct)))
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()
        rows = []
        tested_count = 0
        if strategy == "pullback":
            # Focused SPY research grid. The previous 6,480-combination grid was
            # too expensive for a synchronous Render request and could time out,
            # returning an HTML 502 page to the browser instead of JSON.
            # Stage 1 identifies the useful entry-quality regime; Stage 2 refines
            # RR/stop/cooldown around the best Stage-1 candidates.
            stage1 = list(itertools.product(
                [0.00075, 0.0010, 0.00125, 0.0015],
                [0.05, 0.10, 0.15, 0.20],
                [0.20, 0.30, 0.40],
            ))
            stage1_rows = []
            for min_atr, gap, buf in stage1:
                cfg = PullbackV2Config(
                    min_atr_pct=min_atr,
                    min_ema_gap_atr=gap,
                    breakout_buffer_atr=buf,
                    min_body_atr=0.40,
                    rr=2.0,
                    cooldown_bars=8,
                    atr_stop=1.2,
                )
                a = pullback_v2_backtest(train, cfg)["metrics"]
                b = pullback_v2_backtest(test, cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test:
                    continue
                ap, bp = float(a["profit_factor"] or 0), float(b["profit_factor"] or 0)
                stage1_rows.append({
                    "min_atr_pct":min_atr, "min_ema_gap_atr":gap,
                    "breakout_buffer_atr":buf, "min_body_atr":0.40,
                    "rr":2.0, "cooldown_bars":8, "atr_stop":1.2,
                    "train_trades":a["num_trades"], "train_pf":ap,
                    "train_expectancy_R":a["expectancy_R"],
                    "test_trades":b["num_trades"], "test_pf":bp,
                    "test_expectancy_R":b["expectancy_R"],
                    "test_total_R":b["total_R"], "test_return_pct":b["total_return_pct"],
                    "test_max_dd_pct":b["max_drawdown_pct"],
                    "robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])
                })

            stage1_rows.sort(key=lambda x:(x["robust_score"],x["test_pf"],x["test_total_R"]), reverse=True)
            seeds = stage1_rows[:8]
            tested_count = len(stage1) + (len(seeds) * 3 * 3 * 3 * 3)

            # Refine only around the strongest entry-quality regimes.
            # In batched mode each request owns a disjoint subset of the eight\n            # Stage-1 seeds. This keeps each worker well below the memory peak.\n            seeds_for_batch = seeds if batch_total == 1 else [\n                seed for idx, seed in enumerate(seeds) if idx % batch_total == batch_index\n            ]\n            stage2 = []\n            for seed in seeds_for_batch:
                for body, rr, cooldown, atr_stop in itertools.product(
                    [0.30, 0.40, 0.50],
                    [1.8, 2.0, 2.2],
                    [6, 8, 12],
                    [1.0, 1.2, 1.5],
                ):
                    stage2.append((seed, body, rr, cooldown, atr_stop))

            for seed, body, rr, cooldown, atr_stop in stage2:
                cfg = PullbackV2Config(
                    min_atr_pct=seed["min_atr_pct"],
                    min_ema_gap_atr=seed["min_ema_gap_atr"],
                    breakout_buffer_atr=seed["breakout_buffer_atr"],
                    min_body_atr=body, rr=rr,
                    cooldown_bars=cooldown, atr_stop=atr_stop,
                )
                a = pullback_v2_backtest(train, cfg)["metrics"]
                b = pullback_v2_backtest(test, cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test:
                    continue
                ap, bp = float(a["profit_factor"] or 0), float(b["profit_factor"] or 0)
                rows.append({
                    "min_atr_pct":seed["min_atr_pct"],
                    "min_ema_gap_atr":seed["min_ema_gap_atr"],
                    "breakout_buffer_atr":seed["breakout_buffer_atr"],
                    "min_body_atr":body, "rr":rr,
                    "cooldown_bars":cooldown, "atr_stop":atr_stop,
                    "train_trades":a["num_trades"], "train_pf":ap,
                    "train_expectancy_R":a["expectancy_R"],
                    "test_trades":b["num_trades"], "test_pf":bp,
                    "test_expectancy_R":b["expectancy_R"],
                    "test_total_R":b["total_R"], "test_return_pct":b["total_return_pct"],
                    "test_max_dd_pct":b["max_drawdown_pct"],
                    "robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])
                })

            # Stage 3: structural sensitivity around the strongest Stage-2
            # candidates. We deliberately use coordinate-style tests instead
            # of a full Cartesian grid to keep Render runtime bounded.
            # These are important SPY-specific levers exposed by Pullback V2:
            # EMA structure, pullback depth, breakout lookback and session.
            stage3_seeds = sorted(\n                rows,\n                key=lambda x: (x["robust_score"], x["test_pf"], x["test_total_R"]),\n                reverse=True,\n            )[:2 if batch_total > 1 else 8]\n            stage3 = []\n            tested_count += len(stage3_seeds) * 13
            for seed in stage3_seeds:
                base = {
                    "ema_fast": 50,
                    "ema_slow": 200,
                    "pullback_bars": 3,
                    "breakout_lookback": 5,
                    "session_start_utc": 13,
                    "session_end_utc": 21,
                }
                # One structural dimension at a time, holding the seed's
                # already-optimized entry/exit quality parameters constant.
                candidates = []
                for ef, es in [(50, 200), (30, 100), (20, 50)]:
                    candidates.append({**base, "ema_fast": ef, "ema_slow": es, "stage3_axis": "ema"})
                for pb in [2, 3, 4]:
                    candidates.append({**base, "pullback_bars": pb, "stage3_axis": "pullback_bars"})
                for lookback in [4, 5, 6]:
                    candidates.append({**base, "breakout_lookback": lookback, "stage3_axis": "breakout_lookback"})
                for start_utc, end_utc in [(13, 20), (13, 21), (14, 20), (14, 21)]:
                    candidates.append({
                        **base,
                        "session_start_utc": start_utc,
                        "session_end_utc": end_utc,
                        "stage3_axis": "session",
                    })

                for structural in candidates:
                    cfg = PullbackV2Config(
                        ema_fast=structural["ema_fast"],
                        ema_slow=structural["ema_slow"],
                        pullback_bars=structural["pullback_bars"],
                        breakout_lookback=structural["breakout_lookback"],
                        session_start_utc=structural["session_start_utc"],
                        session_end_utc=structural["session_end_utc"],
                        min_atr_pct=seed["min_atr_pct"],
                        min_ema_gap_atr=seed["min_ema_gap_atr"],
                        breakout_buffer_atr=seed["breakout_buffer_atr"],
                        min_body_atr=seed["min_body_atr"],
                        rr=seed["rr"],
                        cooldown_bars=seed["cooldown_bars"],
                        atr_stop=seed["atr_stop"],
                    )
                    a = pullback_v2_backtest(train, cfg)["metrics"]
                    b = pullback_v2_backtest(test, cfg)["metrics"]
                    if a["num_trades"] < min_train or b["num_trades"] < min_test:
                        continue
                    ap, bp = float(a["profit_factor"] or 0), float(b["profit_factor"] or 0)
                    rows.append({
                        "ema_fast": structural["ema_fast"],
                        "ema_slow": structural["ema_slow"],
                        "pullback_bars": structural["pullback_bars"],
                        "breakout_lookback": structural["breakout_lookback"],
                        "session_start_utc": structural["session_start_utc"],
                        "session_end_utc": structural["session_end_utc"],
                        "min_atr_pct": seed["min_atr_pct"],
                        "min_ema_gap_atr": seed["min_ema_gap_atr"],
                        "breakout_buffer_atr": seed["breakout_buffer_atr"],
                        "min_body_atr": seed["min_body_atr"],
                        "rr": seed["rr"],
                        "cooldown_bars": seed["cooldown_bars"],
                        "atr_stop": seed["atr_stop"],
                        "train_trades": a["num_trades"],
                        "train_pf": ap,
                        "train_expectancy_R": a["expectancy_R"],
                        "train_total_R": a["total_R"],
                        "test_trades": b["num_trades"],
                        "test_pf": bp,
                        "test_expectancy_R": b["expectancy_R"],
                        "test_total_R": b["total_R"],
                        "test_return_pct": b["total_return_pct"],
                        "test_max_dd_pct": b["max_drawdown_pct"],
                        "structural_stage": structural["stage3_axis"],
                        # Return-aware but still robustness-first. PF must
                        # survive train/test; total R gets a modest weight.
                        "robust_score": (
                            min(ap, bp)
                            + 0.35 * float(b["expectancy_R"])
                            + 0.005 * max(float(b["total_R"]), 0.0)
                        ),
                    })

        elif strategy == "ema":
            grid = list(itertools.product([0.05,0.10,0.20], [0.05,0.10,0.20], [2.0,2.5,3.0], [1,2]))
            for gap, rej, rr, sep in grid:
                cfg = EMARetestV2Config(min_gap_atr=gap,min_rejection_body_atr=rej,rr=rr,separation_bars=sep)
                a = ema_v2_backtest(train,cfg)["metrics"]; b = ema_v2_backtest(test,cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test: continue
                ap,bp=float(a["profit_factor"] or 0),float(b["profit_factor"] or 0)
                rows.append({"min_gap_atr":gap,"min_rejection_body_atr":rej,"rr":rr,"separation_bars":sep,"train_trades":a["num_trades"],"train_pf":ap,"train_expectancy_R":a["expectancy_R"],"test_trades":b["num_trades"],"test_pf":bp,"test_expectancy_R":b["expectancy_R"],"test_total_R":b["total_R"],"test_return_pct":b["total_return_pct"],"test_max_dd_pct":b["max_drawdown_pct"],"robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])})
        elif strategy == "trend_ribbon":
            grid = list(itertools.product([0.0, 0.75, 1.0, 1.25, 2.0], [0.05, 0.08, 0.10, 0.12], [0.75, 1.0, 1.5, 2.0, 3.0]))
            for max_body, slope, max_stop in grid:
                cfg = TrendTargetRibbonConfig(max_entry_body_atr=max_body, slope_min=slope, max_stop_atr=max_stop)
                a = trend_ribbon_backtest(train, cfg)["metrics"]; b = trend_ribbon_backtest(test, cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test: continue
                ap, bp = float(a["profit_factor"] or 0), float(b["profit_factor"] or 0)
                rows.append({"max_entry_body_atr":max_body,"slope_min":slope,"max_stop_atr":max_stop,
                             "train_trades":a["num_trades"],"train_pf":ap,"train_expectancy_R":a["expectancy_R"],
                             "test_trades":b["num_trades"],"test_pf":bp,"test_expectancy_R":b["expectancy_R"],
                             "test_total_R":b["total_R"],"test_return_pct":b["total_return_pct"],
                             "test_max_dd_pct":b["max_drawdown_pct"],
                             "robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])})
        else:
            return jsonify({"error":"Optimizer supports Pullback V2, EMA 20/50 V2 and Trend Ribbon V2."}),400
        if strategy in ("ema", "trend_ribbon"):
            tested_count = len(grid)
        if strategy == "pullback":
            # Rank all Pullback results with the same robustness + return score.
            for row in rows:
                row["robust_score"] = (
                    min(float(row.get("train_pf") or 0), float(row.get("test_pf") or 0))
                    + 0.35 * float(row.get("test_expectancy_R") or 0)
                    + 0.005 * max(float(row.get("test_total_R") or 0), 0.0)
                )
        rows.sort(key=lambda x:(x["robust_score"],x["test_pf"],x["test_total_R"]),reverse=True)
        return jsonify(_safe({"strategy":strategy,"symbol":symbol,"data_source":source,"bars":len(df),"train_bars":len(train),"test_bars":len(test),"train_pct":train_pct,"tested":tested_count,"passed":len(rows),"results":rows[:25],"note":"Chronological train/test research only; not a future-performance guarantee."}))
    except Exception as exc:
        return jsonify({"error":f"{type(exc).__name__}: {exc}"}),500


@bp.route("/api/strategy-lab/export-json", methods=["POST"])
def export_strategy_lab_json():
    """Export one complete AI research package: strategies, trades, Alpaca history and multi-timeframe OHLCV."""
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "SPY")).strip().upper()
    selected = body.get("timeframes") or ["5m", "15m", "1h", "1d"]
    if isinstance(selected, str):
        selected = [selected]
    aliases = {"5m":"5m","15m":"15m","1h":"1h","1d":"1d","1D":"1d","1w":"1w","1W":"1w"}
    timeframes = list(dict.fromkeys(aliases[str(x).strip()] for x in selected if str(x).strip() in aliases))
    if not timeframes:
        return jsonify({"error":"Select at least one timeframe."}), 400
    # Export window is calendar-day based. 5m history is the primary
    # validation series; higher timeframes are derived from it when practical.
    days = max(1, min(770, int(body.get("days", 365))))
    bars = max(100, min(60000, int(body.get("bars", min(30000, days * 78)))))
    strategies = body.get("strategies", []) or []
    optimizer = body.get("optimizer") or {}
    run_ids = [str(x.get("run_id","")).strip() for x in strategies if isinstance(x, dict) and x.get("run_id")]
    run_cache = {rid: _RUN_DATA.get(rid) for rid in run_ids}
    exported = datetime.now(timezone.utc).isoformat()

    def market_frame(tf):
        z = None
        for rid, saved in run_cache.items():
            if saved and str(saved.get("native_tf","")).lower() == tf:
                z = saved["df"].tail(bars)
                break
        if z is None:
            # Fetch enough 5m history for the requested calendar window, then
            # derive 15m/1h/1d/1w context from the same continuous series.
            if tf == "5m":
                raw = alpaca_get_bars(symbol, "5Min", limit=bars)
                z = _bars_to_df(raw, symbol, tf)
            else:
                raw = alpaca_get_bars(symbol, "5Min", limit=bars)
                base = _bars_to_df(raw, symbol, "5m")
                rule = {"15m":"15min","1h":"1h","1d":"1D","1w":"W-MON"}[tf]
                z = base.resample(rule, label="left", closed="left").agg({
                    "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
                }).dropna()
        return [{
            "timestamp": ts.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        } for ts, row in z.tail(bars).iterrows()]

    try:
        sync_error = None
        try:
            sync_alpaca_orders(days=30)
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"

        package = {
            "schema_version": "strategy-lab-v3-master-research-1",
            "export_version": "V3",
            "exported_at_utc": exported,
            "symbol": symbol,
            "timeframes": timeframes,
            "days": days,
            "bars_per_timeframe": bars,
            "strategies": strategies,
            "optimization": optimizer,
            "alpaca": {
                "provider": "Alpaca",
                "synced_days": 30,
                "sync_error": sync_error,
                "closed_metrics": get_closed_summary(),
                "order_log": get_all_trades(),
                "closed_positions": get_all_closed_positions(),
            },
            "market_data": {tf: market_frame(tf) for tf in timeframes},
            "contents": {
                "strategies": "Complete Strategy Lab run payloads including metrics, every returned trade, signals/diagnostics and P&L curve.",
                "optimization": "Every optimizer configuration returned by all memory-safe batches, including train/test metrics and robustness score.",
                "alpaca": "Actual broker execution/order history and locally reconstructed closed-position P&L.",
                "market_data": "OHLCV market data for every selected timeframe, kept separate from strategy/backtest trades."
            },
            "analysis_ready": {
                "trade_to_market_mapping": "Use strategy trade timestamps against the native and context timeframe OHLCV.",
                "profit_factor": "Analyze gross profit versus gross loss from individual strategy trades.",
                "return": "Use strategy metrics and P&L curves, then validate changes chronologically.",
                "robustness": "Compare patterns across timeframes, regimes and independent train/test periods."
            },
            "note": "Master AI research export. Strategy/backtest trades and actual Alpaca trades are intentionally kept in separate sections."
        }
        return jsonify(_safe(package))
    except Exception as exc:
        return jsonify({"error":f"{type(exc).__name__}: {exc}"}), 500


@bp.route("/api/strategy-lab/export-excel", methods=["POST"])
def export_strategy_lab_excel():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "SPY")).strip().upper()
    selected = body.get("timeframes") or ["5m", "15m", "1h", "1d"]
    if isinstance(selected, str): selected = [selected]
    aliases = {"5m":"5m","15m":"15m","1h":"1h","1d":"1d","1D":"1d","1w":"1w","1W":"1w"}
    timeframes = list(dict.fromkeys(aliases[str(x).strip()] for x in selected if str(x).strip() in aliases))
    if not timeframes: return jsonify({"error":"Select at least one timeframe."}), 400
    bars = max(100, min(100000, int(body.get("bars", 5000))))
    strategies = body.get("strategies", []) or []
    optimizer = body.get("optimizer") or {}
    # Each strategy run gets its own run_id. Prefer an exact cached run only
    # when its native timeframe matches the requested sheet; otherwise pull
    # the requested timeframe directly so a 5m run cannot silently masquerade
    # as exact 1D/1W data.
    run_ids = [str(x.get("run_id", "")).strip() for x in strategies if isinstance(x, dict)]
    run_cache = {rid: _RUN_DATA.get(rid) for rid in run_ids if rid}
    exported = datetime.now(timezone.utc).isoformat()

    def frame(tf):
        z = None
        # Prefer an exact native cached run for this timeframe.
        for rid, saved in run_cache.items():
            if not saved:
                continue
            native = str(saved.get("native_tf", "")).lower()
            if native == tf:
                z = saved["df"].tail(bars)
                break

        if z is None:
            mp = {"5m":"5Min","15m":"15Min","1h":"1Hour","1d":"1Day","1w":"1Week"}
            raw = alpaca_get_bars(symbol, mp[tf], limit=bars)
            z = _bars_to_df(raw, symbol, tf)

        return _excel_safe_df(
            z.reset_index().rename(columns={"index": "timestamp"})
        )

    try:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            pd.DataFrame([{
                "symbol": symbol,
                "exported_at_utc": exported,
                "run_ids": ", ".join(run_ids),
                "timeframes": ", ".join(timeframes),
                "trade_rows": sum(len(x.get("trades") or []) for x in strategies if isinstance(x, dict)),
                "note": "Market sheets use exact cached native run data when available; otherwise direct Alpaca data for the requested timeframe."
            }]).to_excel(writer, sheet_name="README", index=False)
            for tf in timeframes:
                frame(tf).to_excel(writer, sheet_name=tf.upper(), index=False)
            if optimizer and optimizer.get("results"):
                pd.DataFrame(optimizer.get("results") or []).to_excel(writer, sheet_name="Optimizer_All", index=False)
            trades=[]
            for item in strategies:
                if isinstance(item,dict):
                    for t in item.get("trades") or []:
                        if isinstance(t,dict):
                            row={
                                "strategy": item.get("strategy") or item.get("name") or "",
                                "symbol": symbol,
                                "run_id": item.get("run_id", ""),
                            }
                            row.update(t); trades.append(row)
            _excel_safe_df(pd.DataFrame(trades or [{"strategy":"","symbol":symbol,"note":"No closed trades"}])).to_excel(writer, sheet_name="Trades", index=False)
        out.seek(0)
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return send_file(out, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name=f"{symbol}_strategy_research_{stamp}.xlsx")
    except Exception as exc:
        return jsonify({"error":f"{type(exc).__name__}: {exc}"}), 500

@bp.route("/api/strategy-lab/export", methods=["POST"])
def export_strategy_lab():
    """Export one or many Alpaca OHLCV timeframes plus completed V2 trades."""
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "GLD")).strip().upper()
    selected = body.get("timeframes")
    if not selected:
        selected = [str(body.get("timeframe", "5m")).strip()]
    if isinstance(selected, str):
        selected = [selected]
    aliases = {"5m":"5m","15m":"15m","1h":"1h","1d":"1d","1D":"1d","1w":"1w","1W":"1w"}
    timeframes = []
    for tf in selected:
        key = aliases.get(str(tf).strip())
        if key and key not in timeframes:
            timeframes.append(key)
    if not timeframes:
        return jsonify({"error": "Select at least one supported timeframe: 5m, 15m, 1H, 1D, 1W"}), 400
    bars = max(100, min(100000, int(body.get("bars", 5000))))
    strategies = body.get("strategies", []) or []
    run_id = str(body.get("run_id", "")).strip()
    saved = _RUN_DATA.get(run_id)
    exported = datetime.now(timezone.utc).isoformat()

    def rows_for_tf(tf):
        raw = None
        if saved:
            saved_df = saved["df"]
            # Preserve the exact run bars for the native timeframe. Derived
            # timeframes are aggregated from those exact bars when possible.
            native = str(saved.get("native_tf", "")).lower()
            if tf == native:
                z = saved_df
            elif tf == "15m":
                z = saved_df.resample("15min", label="left", closed="left").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            elif tf == "1h":
                z = saved_df.resample("1h", label="left", closed="left").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            elif tf == "1d":
                z = saved_df.resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            elif tf == "1w":
                z = saved_df.resample("W-MON", label="left", closed="left").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            else:
                z = None
            if z is not None:
                z = z.tail(bars)
                raw = [{"t":ts.isoformat(),"o":float(row.Open),"h":float(row.High),"l":float(row.Low),"c":float(row.Close),"v":float(row.Volume)} for ts,row in z.iterrows()]
        if raw is None:
            tf_map = {"5m":"5Min","15m":"15Min","1h":"1Hour","1d":"1Day","1w":"1Week"}
            raw = alpaca_get_bars(symbol, tf_map[tf], limit=bars)
            raw = raw[-bars:]
        if not raw:
            raise ValueError(f"No Alpaca {tf} data returned for {symbol}")
        rows = [{
            "exported_at_utc":exported,"record_type":"market_data","strategy":"",
            "symbol":symbol,"timeframe":tf,"timestamp":b.get("t",""),
            "open":b.get("o"),"high":b.get("h"),"low":b.get("l"),"close":b.get("c"),"volume":b.get("v")
        } for b in raw]
        for item in strategies:
            if not isinstance(item,dict): continue
            name=str(item.get("strategy") or item.get("name") or "").strip()
            for trade in item.get("trades") or []:
                if not isinstance(trade,dict): continue
                row={"exported_at_utc":exported,"record_type":"trade","strategy":name,"symbol":symbol,"timeframe":tf}
                for key,value in trade.items():
                    row[str(key)] = json.dumps(value,ensure_ascii=False,separators=(",",":")) if isinstance(value,(dict,list)) else value
                rows.append(row)
        return rows

    try:
        files = {}
        base=["exported_at_utc","record_type","strategy","symbol","timeframe","timestamp","open","high","low","close","volume"]
        for tf in timeframes:
            rows=rows_for_tf(tf)
            extra=sorted({k for row in rows for k in row}-set(base))
            output=io.StringIO()
            writer=csv.DictWriter(output,fieldnames=base+extra,extrasaction="ignore")
            writer.writeheader();writer.writerows(rows)
            files[f"{symbol}_strategy_lab_v2_{tf}.csv"]=output.getvalue().encode("utf-8-sig")
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if len(files)==1:
            filename,payload=next(iter(files.items()))
            return send_file(io.BytesIO(payload),mimetype="text/csv; charset=utf-8",as_attachment=True,
                             download_name=filename.replace(".csv",f"_{stamp}.csv"))
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
            for filename,payload in files.items():
                zf.writestr(filename,payload)
        archive.seek(0)
        return send_file(archive,mimetype="application/zip",as_attachment=True,
                         download_name=f"{symbol}_strategy_lab_v2_multitimeframe_{stamp}.zip")
    except Exception as exc:
        return jsonify({"error":f"{type(exc).__name__}: {exc}"}),500

@bp.route("/api/xauusd-research-v2/<strategy>", methods=["POST"])
def run(strategy):
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    strategy = str(strategy).strip().lower()
    symbol = str(body.get("symbol", "GLD")).strip().upper()

    try:
        if strategy == "triple":
            daily_bars = int(body.get("daily_bars", 1000))
            df, source = _daily(symbol, daily_bars)
            cfg = _cfg(MultiHorizonRSIConfig, body.get("config"))
            result = backtest_multi_rsi(df, cfg)
        elif strategy in ("williams", "cci", "multi_rsi"):
            daily_bars = int(body.get("daily_bars", 1000))
            df, source = _daily(symbol, daily_bars)
            if strategy == "williams":
                cfg = _cfg(WilliamsRConfig, body.get("config"))
                result = backtest_williams_r(df, cfg)
            elif strategy == "cci":
                cfg = _cfg(CCIConfig, body.get("config"))
                result = backtest_cci(df, cfg)
            else:
                cfg = _cfg(MultiHorizonRSIConfig, body.get("config"))
                result = backtest_multi_rsi(df, cfg)
        elif strategy == "pullback":
            bars = int(body.get("n_bars", 5000))
            df, source = _intraday(symbol, bars)
            cfg = _cfg(PullbackV2Config, body.get("config"))
            result = pullback_v2_backtest(df, cfg)
        elif strategy == "ema":
            bars = int(body.get("n_bars", 15600))
            df, source = _intraday(symbol, bars)
            cfg = _cfg(EMARetestV2Config, body.get("config"))
            result = ema_v2_backtest(df, cfg)
        elif strategy in ("trend_ribbon", "trend-target-ribbon", "trend_target_ribbon"):
            bars = int(body.get("n_bars", 15600))
            df, source = _intraday(symbol, bars)
            cfg = _cfg(TrendTargetRibbonConfig, body.get("config"))
            result = trend_ribbon_backtest(df, cfg)
        else:
            return jsonify({"error": f"Unknown V2 strategy: {strategy}"}), 400

        trades = result.get("trades", [])
        if hasattr(trades, "to_dict"):
            trades = trades.to_dict(orient="records")

        signals = result.get("signals")
        if hasattr(signals, "tail"):
            signals = (
                signals.tail(500)
                .reset_index()
                .to_dict(orient="records")
            )

        # Compact P&L/equity curve for the webpage and GitHub-readable JSON.
        equity_curve = result.get("equity_curve")
        pnl_curve = []
        if hasattr(equity_curve, "items") and len(equity_curve):
            points = equity_curve.astype(float)
            step = max(1, int(len(points) / 300))
            sampled = points.iloc[::step]
            if sampled.index[-1] != points.index[-1]:
                sampled = pd.concat([sampled, points.iloc[[-1]]])
            initial = float(result.get("metrics", {}).get("initial_equity", 10000.0))
            for ts, eq in sampled.items():
                pnl_curve.append({
                    "timestamp": ts.isoformat(),
                    "equity": float(eq),
                    "pnl": float(eq - initial),
                    "return_pct": float((eq / initial - 1.0) * 100.0) if initial else 0.0,
                })

        run_id = uuid.uuid4().hex
        _RUN_DATA[run_id] = {"df":df.copy(),"symbol":symbol,"source":source,"created":time.time(),"native_tf":("1d" if strategy in ("triple","williams","cci","multi_rsi") else "5m")}
        if len(_RUN_DATA) > 20:
            for key,_ in sorted(_RUN_DATA.items(),key=lambda kv:kv[1]["created"])[:5]:
                _RUN_DATA.pop(key,None)
        return jsonify(_safe({
            "run_id": run_id,
            "strategy": strategy + " V2",
            "symbol": symbol,
            "data_source": source,
            "bars": len(df),
            "data_start": df.index.min().isoformat(),
            "data_end": df.index.max().isoformat(),
            "metrics": result.get("metrics", {}),
            "trades": trades,
            "signals": signals,
            "diagnostics": result.get("diagnostics", {}),
            "pnl_curve": pnl_curve,
        }))
    except Exception as exc:
        # Always return JSON so the V2 page can display the real backend
        # failure instead of showing a generic "not running" message.
        return jsonify({
            "error": f"{type(exc).__name__}: {exc}",
            "strategy": strategy + " V2",
            "symbol": symbol,
        }), 500
