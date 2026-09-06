"""
research/strategy_signals.py
A specific, well-documented technical strategy (not a black box): 9/21 EMA
crossover, confirmed by the 50 EMA as a trend filter, with RSI and intraday
VWAP used to spot momentum fading on the way out. This combination is one
of the more commonly backtested EMA approaches for US equities/ETFs (9/21
crossover as the timing signal, 50 EMA as the trend confirmation, RSI/VWAP
divergence as an early exit warning).

This is NOT a guarantee of profit -- it's a rules-based heuristic. See
research/backtester.py to check how this exact ruleset would have performed
historically on a given symbol before trusting it live.

Signals produced:
  - BUY SETUP:  9 EMA crosses above 21 EMA, price above the 50 EMA, ideally
                also reclaiming/holding above today's VWAP.
  - TAKE-PROFIT / REVERSAL WARNING: price closing back below EMA9 (early)
                or EMA21 (stronger) after a run-up. Confirmed further if the
                stock was extended well above VWAP and has now lost it, or
                if RSI is cooling from overbought.
  - VWAP REVERSAL WATCH: even without an EMA break yet, losing VWAP after
                being meaningfully extended above it intraday is a classic
                early same-day warning that a move is running out of steam.
"""
from core.logger import get_logger
from core.market_data import get_bars
from core.analyzer import calc_rsi

logger = get_logger(__name__)


def _ema_series(closes: list, span: int) -> list:
    """Standard EMA, seeded with the first close (matches the rest of this
    codebase's convention, e.g. the EMA Signal Summary table in the UI)."""
    if not closes:
        return []
    k = 2 / (span + 1)
    s = closes[0]
    out = [s]
    for c in closes[1:]:
        s = c * k + s * (1 - k)
        out.append(s)
    return out


def _intraday_vwap_series(bars: list) -> list:
    """Same VWAP formula used by dashboard/routes.py's chart_data endpoint
    (cumulative typical price * volume / cumulative volume), reset each new
    session -- kept independent here so this module has no dependency on
    dashboard code."""
    out, cum_pv, cum_vol, last_day = [], 0.0, 0.0, None
    for b in bars:
        o, h, l, c, v = b.get("o"), b.get("h"), b.get("l"), b.get("c"), b.get("v") or 0
        if None in (o, h, l, c):
            out.append(None)
            continue
        day = (b.get("t") or "")[:10]
        if day != last_day:
            cum_pv, cum_vol, last_day = 0.0, 0.0, day
        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_vol += v
        out.append(cum_pv / cum_vol if cum_vol else c)
    return out


def _get_vwap_signal(symbol: str) -> dict:
    """Fetches today's intraday bars separately (VWAP is a same-day concept,
    not something you compute off 6mo daily bars). Fails soft -- returns
    all-None if intraday data isn't available (e.g. market closed, or
    demo-data fallback), so the caller can just skip VWAP confirmation."""
    try:
        bars = get_bars(symbol, "1D")  # today's 15m bars, per PERIOD_CONFIG
        if not bars or any(b.get("source") == "demo" for b in bars):
            return {"vwap_now": None, "price_vs_vwap_pct": None, "lost_vwap": False, "extended_above_vwap": False}
        vwap = _intraday_vwap_series(bars)
        closes = [b.get("c") for b in bars]
        if len(vwap) < 6 or vwap[-1] is None:
            return {"vwap_now": None, "price_vs_vwap_pct": None, "lost_vwap": False, "extended_above_vwap": False}

        price = closes[-1]
        vwap_now = vwap[-1]
        pct = (price - vwap_now) / vwap_now * 100 if vwap_now else None

        # Was it meaningfully extended above VWAP in the last few bars, and
        # has it now dropped back under? That sequence -- extended, then
        # lost VWAP -- is the actual "reversal" signal, not just "below
        # VWAP" alone (a stock can trade under VWAP all day with no story).
        recent_pcts = [
            (closes[i] - vwap[i]) / vwap[i] * 100
            for i in range(max(0, len(vwap) - 6), len(vwap))
            if vwap[i] and closes[i] is not None
        ]
        was_extended = any(p >= 1.5 for p in recent_pcts[:-1]) if len(recent_pcts) > 1 else False
        lost_vwap = was_extended and pct is not None and pct < 0

        return {
            "vwap_now": round(vwap_now, 4), "price_vs_vwap_pct": round(pct, 2) if pct is not None else None,
            "lost_vwap": lost_vwap, "extended_above_vwap": pct is not None and pct >= 1.5,
        }
    except Exception as e:
        logger.warning(f"strategy_signals VWAP step failed for {symbol}: {e}")
        return {"vwap_now": None, "price_vs_vwap_pct": None, "lost_vwap": False, "extended_above_vwap": False}


def get_strategy_signal(symbol: str, period: str = "6mo", unrealized_pct: float = None) -> dict:
    """
    Computes the 9/21/50 EMA + VWAP + RSI strategy signal for one symbol.
    `unrealized_pct` (optional): if this symbol is a held position, pass its
    current unrealized % gain so a profitable-and-fading position can be
    flagged as a take-profit candidate specifically (vs. a fresh short).
    """
    symbol = symbol.upper()
    bars = get_bars(symbol, period)
    closes = [b["c"] for b in bars if b.get("c") is not None]

    if len(closes) < 55:
        return {"symbol": symbol, "label": "Insufficient Data",
                "reason": f"Only {len(closes)} bars — need 55+ for a stable 50 EMA.", "action": "none"}

    ema9  = _ema_series(closes, 9)
    ema21 = _ema_series(closes, 21)
    ema50 = _ema_series(closes, 50)

    price = closes[-1]
    bullish_cross = ema9[-2] <= ema21[-2] and ema9[-1] > ema21[-1]
    bearish_cross = ema9[-2] >= ema21[-2] and ema9[-1] < ema21[-1]
    trend_up   = price > ema50[-1]
    trend_down = price < ema50[-1]
    price_below_ema9  = price < ema9[-1]
    price_below_ema21 = price < ema21[-1]

    rsi_now  = calc_rsi(closes)
    rsi_prev = calc_rsi(closes[:-1])
    rsi_cooling = (rsi_prev is not None and rsi_now is not None
                   and rsi_prev >= 70 and rsi_now < rsi_prev and rsi_now < 70)

    vwap_info = _get_vwap_signal(symbol)

    label, reason, action = "Hold / No Signal", "No crossover or reversal condition met.", "none"

    if bullish_cross and trend_up:
        label = "🟢 Buy Setup"
        reason = "9 EMA crossed above 21 EMA while price holds above the 50 EMA (uptrend confirmed)."
        action = "buy_setup"
        if vwap_info["vwap_now"] and price > vwap_info["vwap_now"]:
            reason += " Also holding above today's VWAP — added confirmation."
    elif bearish_cross and trend_down:
        label = "🔴 Sell/Short Setup"
        reason = "9 EMA crossed below 21 EMA while price is under the 50 EMA (downtrend confirmed)."
        action = "sell_setup"
    elif price_below_ema21:
        label = "⚠️ Reversal Confirmed"
        reason = "Price has closed back below the 21 EMA — the prior uptrend's momentum is broken."
        action = "take_profit_strong"
    elif price_below_ema9:
        label = "🟡 Early Reversal Warning"
        reason = "Price closed below the 9 EMA — short-term momentum is fading, worth watching closely."
        action = "take_profit_watch"
    elif vwap_info["lost_vwap"]:
        # No EMA break yet, but this is still a real, earlier warning sign:
        # extended well above VWAP intraday, then lost it.
        label = "🟡 VWAP Reversal Watch"
        reason = (f"Was extended above VWAP intraday and has now dropped back under "
                  f"(currently {vwap_info['price_vs_vwap_pct']}% vs VWAP) — early sign of buyers stepping back.")
        action = "take_profit_watch"

    if rsi_cooling:
        reason += f" RSI cooling from overbought ({rsi_prev:.0f}→{rsi_now:.0f}) adds confirmation."

    if vwap_info["lost_vwap"] and action in ("take_profit_watch", "take_profit_strong"):
        reason += f" Also lost VWAP support intraday (now {vwap_info['price_vs_vwap_pct']}% vs VWAP)."
        if action == "take_profit_watch":
            action = "take_profit_strong"  # losing VWAP too upgrades conviction
            label = "⚠️ Reversal Confirmed (VWAP + EMA)"

    # Reframe as a take-profit-specific call if this is a position already in profit
    if unrealized_pct is not None and unrealized_pct > 5 and action in ("take_profit_watch", "take_profit_strong"):
        label = f"💰 Take-Profit Candidate ({action.split('_')[-1]})"

    return {
        "symbol": symbol, "price": round(price, 4),
        "ema9": round(ema9[-1], 4), "ema21": round(ema21[-1], 4), "ema50": round(ema50[-1], 4),
        "rsi": rsi_now, "bullish_cross": bullish_cross, "bearish_cross": bearish_cross,
        "trend_up": trend_up, "label": label, "reason": reason, "action": action,
        "vwap_now": vwap_info["vwap_now"], "price_vs_vwap_pct": vwap_info["price_vs_vwap_pct"],
    }
