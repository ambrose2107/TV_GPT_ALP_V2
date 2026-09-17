"""
visualize.py
Charts for the MTF Fibonacci strategy report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fibonacci import FIB_LEVELS, fib_levels_for_leg


def plot_overview(df5: pd.DataFrame, trades: pd.DataFrame, equity_curve: pd.Series, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2.5, 1, 1]}, sharex=True)

    ax = axes[0]
    ax.plot(df5.index, df5["Close"], color="#888", lw=0.6)
    if not trades.empty:
        longs = trades[trades["side"] == "LONG"]
        shorts = trades[trades["side"] == "SHORT"]
        ax.scatter(longs["entry_time"], longs["entry_price"], marker="^", color="lime", s=35,
                  zorder=5, edgecolors="black", label="Long entry")
        ax.scatter(shorts["entry_time"], shorts["entry_price"], marker="v", color="red", s=35,
                  zorder=5, edgecolors="black", label="Short entry")
    ax.set_title("XAUUSD 5m - Price & Entries (MTF Fibonacci Strategy)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(equity_curve.index, equity_curve.values, color="#2ca02c", lw=1.0)
    ax2.set_title("Equity Curve")
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


def plot_fib_and_trade_detail(df5: pd.DataFrame, df15: pd.DataFrame, trades: pd.DataFrame, out_path: str,
                               trade_index: int = -1, bars_padding_5m: int = 40):
    """
    Zooms into ONE trade (default: most recent) and shows:
      - ALL Fibonacci levels for the 15m impulse that produced the setup
      - the primary 0.618-0.786 zone shaded
      - the SL (red/gold-after-breakeven) and all 4 structural TP levels
        (TP3/TP4 highlighted), drawn from entry to exit
    """
    if trades.empty:
        return None
    t = trades.iloc[trade_index]

    entry_time = t["entry_time"]
    exit_time = t["exit_time"] if pd.notna(t["exit_time"]) else df5.index[-1]

    # find the 15m impulse active at entry time
    row15 = df15.loc[:entry_time].iloc[-1]
    direction = "bullish" if t["side"] == "LONG" else "bearish"
    imp_low = row15["impulse_low"]
    imp_high = row15["impulse_high"]
    if pd.isna(imp_low) or pd.isna(imp_high):
        levels = None
    else:
        levels = fib_levels_for_leg(imp_low, imp_high, direction)

    start_idx = df5.index.get_indexer([entry_time], method="nearest")[0]
    end_idx = df5.index.get_indexer([exit_time], method="nearest")[0]
    lo_i = max(0, start_idx - bars_padding_5m)
    hi_i = min(len(df5) - 1, end_idx + bars_padding_5m)
    window = df5.iloc[lo_i:hi_i + 1]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(window.index, window["Close"], color="#444", lw=0.8, zorder=1)

    if levels is not None:
        cmap = plt.get_cmap("RdYlGn_r" if direction == "bullish" else "RdYlGn")
        for i, pct in enumerate(FIB_LEVELS):
            price = levels[pct]
            color = cmap(i / (len(FIB_LEVELS) - 1))
            lw = 1.8 if pct in (0.618, 0.705, 0.786) else 0.8
            ax.axhline(price, color=color, lw=lw, linestyle="--", alpha=0.7, zorder=2)
            ax.text(window.index[-1], price, f" {pct*100:.1f}%", color=color, fontsize=7,
                    va="center", fontweight="bold")
        zone_lo, zone_hi = sorted([levels[0.618], levels[0.786]])
        ax.axhspan(zone_lo, zone_hi, color="purple", alpha=0.08, zorder=0)

    ax.scatter([entry_time], [t["entry_price"]], marker="^" if t["side"] == "LONG" else "v",
              color="lime" if t["side"] == "LONG" else "red", s=100, zorder=6, edgecolors="black")

    sl_color = "gold" if t["breakeven_triggered"] else "red"
    ax.hlines(t["sl_initial"], entry_time, exit_time, color=sl_color, lw=2.2, linestyle="--", zorder=4)
    ax.text(exit_time, t["sl_initial"], "  SL" + (" (BE)" if t["breakeven_triggered"] else ""),
            color=sl_color, fontsize=9, va="center", fontweight="bold")

    tp_style = {"tp1": ("#90ee90", 1.2, ":"), "tp2": ("#90ee90", 1.2, ":"),
                "tp3": ("#008080", 2.2, "-"), "tp4": ("#0000ff", 2.2, "-")}
    for key in ["tp1", "tp2", "tp3", "tp4"]:
        val = t[key]
        if pd.notna(val):
            color, lw, style = tp_style[key]
            ax.hlines(val, entry_time, exit_time, color=color, lw=lw, linestyle=style, zorder=3)
            if key in ("tp3", "tp4"):
                ax.text(exit_time, val, f"  {key.upper()}", color=color, fontsize=9, va="center", fontweight="bold")

    ax.set_title(f"Trade Detail: {t['side']} @ {entry_time} - All Fib Levels + Structural SL/TP "
                 f"(TP3/TP4 highlighted)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_funnel(funnel: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    stages = list(funnel.keys())
    counts = list(funnel.values())
    colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, len(stages)))
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    ax.set_title("Signal Funnel: 15m Bias -> Impulse -> Fib+Confluence -> 5m Sweep+BOS")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
