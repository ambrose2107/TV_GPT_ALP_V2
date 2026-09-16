"""
visualize.py
Chart generation for the backtest report:
  - Overview: price + EMAs + entries, equity curve + drawdown below
  - Trade detail: zoomed view of the most recent N trades with SL, all hit
    TP levels (TP1-4), and the breakeven move drawn as explicit lines
  - Fibonacci levels: the most recent swing marked with ALL standard levels
    (0, 23.6, 38.2, 50, 61.8, 78.6, 100, 127.2, 161.8), not just one zone
  - Zones: supply/demand, FVG, and volume POI shaded over a recent window
  - Funnel: bar chart of the signal funnel (diagnoses "no trades" issues)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from .indicators import fibonacci_levels, FIB_LEVELS


def plot_overview(df: pd.DataFrame, trades: pd.DataFrame, equity_curve: pd.Series,
                   out_path: str, title: str = "XAUUSD"):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2.5, 1, 1]}, sharex=True)

    ax = axes[0]
    ax.plot(df.index, df["Close"], color="#888", lw=0.6, label="Close")
    if "ema_f" in df.columns:
        ax.plot(df.index, df["ema_f"], color="#1f77b4", lw=0.8, label="EMA Fast")
        ax.plot(df.index, df["ema_s"], color="#ff7f0e", lw=0.8, label="EMA Slow")
    if not trades.empty:
        longs = trades[trades["side"] == "LONG"]
        shorts = trades[trades["side"] == "SHORT"]
        ax.scatter(longs["entry_time"], longs["entry_price"], marker="^", color="lime", s=30,
                   zorder=5, label="Long entry", edgecolors="black", linewidths=0.5)
        ax.scatter(shorts["entry_time"], shorts["entry_price"], marker="v", color="red", s=30,
                   zorder=5, label="Short entry", edgecolors="black", linewidths=0.5)
    ax.set_title(f"{title} - Price & Entries")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(equity_curve.index, equity_curve.values, color="#2ca02c", lw=1.0)
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Equity ($)")
    ax2.grid(alpha=0.2)

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max * 100
    ax3 = axes[2]
    ax3.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax3.set_title("Drawdown (%)")
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trade_detail(df: pd.DataFrame, trades: pd.DataFrame, out_path: str,
                       n_recent_trades: int = 5, bars_padding: int = 15):
    """
    Zooms into the most recent N trades and draws, for EACH trade:
      - entry marker
      - SL line (red, turns gold/dashed after breakeven) from entry to exit
      - every TP level that was actually configured (TP1-4), with TP3/TP4
        drawn thicker/brighter as the 'highlighted' levels, per the original
        chart request
    """
    if trades.empty:
        return None

    recent = trades.tail(n_recent_trades).copy()
    start_time = recent["entry_time"].min()
    end_time = recent["exit_time"].max()

    start_idx = df.index.get_indexer([start_time], method="nearest")[0]
    end_idx = df.index.get_indexer([end_time], method="nearest")[0]
    lo = max(0, start_idx - bars_padding)
    hi = min(len(df) - 1, end_idx + bars_padding)
    window = df.iloc[lo:hi + 1]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(window.index, window["Close"], color="#555", lw=0.8, zorder=1)

    tp_colors = {"tp1": ("#90ee90", 1, ":"), "tp2": ("#90ee90", 1, ":"),
                 "tp3": ("#008080", 2, "-"), "tp4": ("#0000ff", 2, "-")}

    for _, t in recent.iterrows():
        et, xt = t["entry_time"], t["exit_time"] if pd.notna(t["exit_time"]) else window.index[-1]
        side_color = "lime" if t["side"] == "LONG" else "red"
        ax.scatter([et], [t["entry_price"]], marker="^" if t["side"] == "LONG" else "v",
                   color=side_color, s=80, zorder=5, edgecolors="black")

        # SL line: red before breakeven, gold after
        sl_color = "gold" if t["breakeven_triggered"] else "red"
        ax.hlines(t["sl_initial"], et, xt, color=sl_color, lw=2, linestyle="--", zorder=3)
        ax.text(xt, t["sl_initial"], "  SL" + (" (BE)" if t["breakeven_triggered"] else ""),
                color=sl_color, fontsize=8, va="center", fontweight="bold")

        for tp_key in ["tp1", "tp2", "tp3", "tp4"]:
            tp_val = t[tp_key]
            if pd.notna(tp_val):
                color, lw, style = tp_colors[tp_key]
                ax.hlines(tp_val, et, xt, color=color, lw=lw, linestyle=style, zorder=2)
                if tp_key in ("tp3", "tp4"):
                    ax.text(xt, tp_val, f"  {tp_key.upper()}", color=color, fontsize=8,
                            va="center", fontweight="bold")

    ax.set_title(f"Trade Detail - Last {len(recent)} Trades (SL/TP lines; TP3-4 highlighted)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fibonacci_levels(df: pd.DataFrame, swing_lookback: int, out_path: str,
                           n_recent_bars: int = 150):
    """
    Marks ALL standard Fibonacci levels (0, 23.6, 38.2, 50, 61.8, 78.6, 100,
    127.2, 161.8) for the most recent swing, over the last n_recent_bars of
    price action - not just the single retracement zone used as a filter.
    """
    window = df.tail(n_recent_bars)
    swing_high = window["High"].max()
    swing_low = window["Low"].min()
    swing_high_time = window["High"].idxmax()
    swing_low_time = window["Low"].idxmin()
    uptrend = swing_low_time < swing_high_time  # leg ran low -> high

    levels = fibonacci_levels(swing_low, swing_high, uptrend)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(window.index, window["Close"], color="#333", lw=0.9, zorder=1)

    cmap = plt.get_cmap("RdYlGn_r" if uptrend else "RdYlGn")
    for i, pct in enumerate(FIB_LEVELS):
        price = levels[pct]
        color = cmap(i / (len(FIB_LEVELS) - 1))
        lw = 1.8 if pct in (0.5, 0.618, 0.786) else 1.0
        ax.axhline(price, color=color, lw=lw, linestyle="--", alpha=0.85, zorder=2)
        ax.text(window.index[-1], price, f"  {pct*100:.1f}%  ({price:.2f})",
                color=color, fontsize=8, va="center", fontweight="bold")

    ax.scatter([swing_high_time], [swing_high], color="red", marker="v", s=100, zorder=5, label="Swing High")
    ax.scatter([swing_low_time], [swing_low], color="green", marker="^", s=100, zorder=5, label="Swing Low")
    ax.set_title(f"Fibonacci Levels - {'Uptrend' if uptrend else 'Downtrend'} Leg "
                 f"(lookback={swing_lookback} bars, all standard levels marked)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_zones(df: pd.DataFrame, confluence_data: dict, out_path: str, n_recent_bars: int = 300):
    """Shades supply/demand, FVG, and volume-POI zones over a recent window."""
    window = df.tail(n_recent_bars)
    w_start = window.index[0]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(window.index, window["Close"], color="#333", lw=0.9, zorder=3)

    sd = confluence_data["sd"]
    fvg = confluence_data["fvg"]
    poi = confluence_data["poi"]

    def _shade(valid_series, lo_series, hi_series, color, label):
        v = valid_series.reindex(window.index).fillna(False)
        if not v.any():
            return
        lo = lo_series.reindex(window.index)
        hi = hi_series.reindex(window.index)
        # find contiguous True runs for cleaner shading
        in_run = False
        run_start = None
        run_lo = run_hi = None
        first_label = True
        for t, is_valid in v.items():
            if is_valid and not in_run:
                in_run, run_start, run_lo, run_hi = True, t, lo[t], hi[t]
            elif is_valid and in_run:
                run_lo, run_hi = min(run_lo, lo[t]), max(run_hi, hi[t])
            elif not is_valid and in_run:
                ax.axhspan(run_lo, run_hi, xmin=(window.index.get_loc(run_start) / len(window)),
                          xmax=(window.index.get_loc(t) / len(window)), color=color, alpha=0.15,
                          label=label if first_label else None)
                first_label = False
                in_run = False
        if in_run:
            ax.axhspan(run_lo, run_hi, xmin=(window.index.get_loc(run_start) / len(window)), xmax=1.0,
                      color=color, alpha=0.15, label=label if first_label else None)

    _shade(sd["demand_valid"], sd["demand_lo"], sd["demand_hi"], "green", "Demand Zone")
    _shade(sd["supply_valid"], sd["supply_lo"], sd["supply_hi"], "red", "Supply Zone")
    _shade(fvg["fvg_bull_valid"], fvg["fvg_bull_lo"], fvg["fvg_bull_hi"], "teal", "Bullish FVG")
    _shade(fvg["fvg_bear_valid"], fvg["fvg_bear_lo"], fvg["fvg_bear_hi"], "orange", "Bearish FVG")
    _shade(poi["poi_valid"], poi["poi_lo"], poi["poi_hi"], "gold", "Volume POI")

    ax.set_title("Institutional Zones (recent window): Supply/Demand, FVG, Volume POI")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_funnel(funnel: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = list(funnel.keys())
    counts = list(funnel.values())
    colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, len(stages)))
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)
    ax.set_title("Signal Funnel - Where Trades Get Filtered Out")
    ax.set_xlabel("Number of qualifying bars")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
