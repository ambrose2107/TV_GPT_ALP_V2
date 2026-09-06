"""
research/backtester.py
Runs the EXACT same rule set as research/strategy_signals.py (9/21 EMA
crossover, 50 EMA trend filter, EMA21 reversal exit) against real historical
daily bars, bar-by-bar, with no look-ahead bias -- a signal computed "as of"
bar i only ever uses bars[0..i], never future data.

This is how you check whether a strategy is worth trusting BEFORE risking
money on it: does it actually beat just holding the stock, over real
history, net of how often it's wrong?

This is not a guarantee of future performance -- past results, even real
ones, don't promise future ones. Markets, regimes and individual stocks
change. Treat this as one more data point, not a certainty.
"""
from datetime import datetime

from core.logger import get_logger
from core.market_data import get_bars

logger = get_logger(__name__)


def _ema_series(closes: list, span: int) -> list:
    if not closes:
        return []
    k = 2 / (span + 1)
    s = closes[0]
    out = [s]
    for c in closes[1:]:
        s = c * k + s * (1 - k)
        out.append(s)
    return out


def backtest_symbol(symbol: str, period: str = "3y") -> dict:
    """
    Simulates the 9/21/50 EMA strategy on real historical bars:
      ENTRY: 9 EMA crosses above 21 EMA AND price > 50 EMA (same rule as
             the live "Buy Setup" signal).
      EXIT:  price closes back below the 21 EMA (same rule as the live
             "Reversal Confirmed" signal), or end of data if still open.
    Only ever long, one position at a time (no pyramiding, no shorts) --
    keeps this directly comparable to "should I have bought and later sold."
    """
    symbol = symbol.upper()
    bars = get_bars(symbol, period)
    closes = [b["c"] for b in bars if b.get("c") is not None]
    dates  = [b["t"] for b in bars if b.get("c") is not None]

    if len(closes) < 60:
        return {"symbol": symbol, "error": f"Only {len(closes)} bars — need 60+ for a meaningful backtest."}

    ema9  = _ema_series(closes, 9)
    ema21 = _ema_series(closes, 21)
    ema50 = _ema_series(closes, 50)

    trades = []
    in_position = False
    entry_price = entry_date = None

    # Start at index 50 so the 50 EMA has had time to become meaningful
    # (an EMA is technically defined from bar 0, but it's noisy/unrepresentative
    # very early on -- same convention used by strategy_signals.py's 55-bar minimum).
    for i in range(50, len(closes)):
        price = closes[i]
        bullish_cross = ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]
        bearish_break  = price < ema21[i]
        trend_up = price > ema50[i]

        if not in_position and bullish_cross and trend_up:
            in_position = True
            entry_price = price
            entry_date = dates[i]
        elif in_position and bearish_break:
            exit_price = price
            ret_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "entry_date": entry_date, "entry_price": round(entry_price, 4),
                "exit_date": dates[i], "exit_price": round(exit_price, 4),
                "return_pct": round(ret_pct, 2),
            })
            in_position = False

    # If still holding at the end of the data, close it out "mark to market"
    # so the stats reflect a real, complete picture (not silently dropped).
    open_trade = None
    if in_position:
        exit_price = closes[-1]
        ret_pct = (exit_price - entry_price) / entry_price * 100
        open_trade = {
            "entry_date": entry_date, "entry_price": round(entry_price, 4),
            "exit_date": dates[-1], "exit_price": round(exit_price, 4),
            "return_pct": round(ret_pct, 2), "status": "still open (marked to last close)",
        }

    all_trades = trades + ([open_trade] if open_trade else [])
    n = len(all_trades)

    if n == 0:
        return {
            "symbol": symbol, "period": period, "total_trades": 0,
            "note": "No completed buy-setup trades in this window under this ruleset.",
            "buy_hold_return_pct": round((closes[-1] - closes[0]) / closes[0] * 100, 2),
        }

    wins = [t for t in all_trades if t["return_pct"] > 0]
    losses = [t for t in all_trades if t["return_pct"] <= 0]
    win_rate = round(len(wins) / n * 100, 1)
    avg_return = round(sum(t["return_pct"] for t in all_trades) / n, 2)
    avg_win = round(sum(t["return_pct"] for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t["return_pct"] for t in losses) / len(losses), 2) if losses else 0

    # Compounded return if you took every signal in sequence (not cash-weighted
    # across overlapping positions -- this strategy is one-at-a-time by design).
    compounded = 1.0
    for t in all_trades:
        compounded *= (1 + t["return_pct"] / 100)
    total_return_pct = round((compounded - 1) * 100, 2)

    buy_hold_return_pct = round((closes[-1] - closes[0]) / closes[0] * 100, 2)

    return {
        "symbol": symbol, "period": period,
        "total_trades": n, "wins": len(wins), "losses": len(losses),
        "win_rate_pct": win_rate, "avg_return_pct": avg_return,
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "total_strategy_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "beat_buy_hold": total_return_pct > buy_hold_return_pct,
        "trades": all_trades,
        "backtested_at": datetime.utcnow().isoformat() + "Z",
    }
