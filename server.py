import asyncio
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timedelta
from pydantic import BaseModel

from core.database import init_db, get_db, Trade, BotStatus
from core.risk_engine import risk_engine
from core.logger import get_logger
from core.strategy_loader import get_strategy_info
from brokers.alpaca_adapter import alpaca_broker
from bot import bot
from backtest import Backtester

log = get_logger("dashboard")
app = FastAPI(title="Trading Bot", version="3.0.0")

if os.path.exists("dashboard/static"):
    app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")


@app.on_event("startup")
async def startup():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    await init_db()
    log.info("=" * 50)
    log.info("Dashboard started")
    try:
        acc = alpaca_broker.get_account()
        log.info(
            f"✓ Alpaca connected | equity=${acc['equity']} | mode={alpaca_broker.mode}"
        )
    except Exception as e:
        log.error(f"✗ Alpaca FAILED: {e}")
    try:
        strats = get_strategy_info()
        log.info(f"✓ Strategies: {[s['name'] for s in strats]}")
    except Exception as e:
        log.error(f"✗ Strategy loader FAILED: {e}")
    log.info("=" * 50)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard/index.html") as f:
        content = f.read()
    response = HTMLResponse(content=content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/account")
async def get_account():
    try:
        return alpaca_broker.get_account()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/positions")
async def get_positions():
    try:
        return alpaca_broker.get_positions()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Trade).order_by(desc(Trade.created_at)).limit(limit)
    )
    trades = result.scalars().all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "total_value": t.total_value,
            "strategy": t.strategy,
            "status": t.status,
            "pnl": t.pnl,
            "created_at": str(t.created_at),
        }
        for t in trades
    ]


@app.get("/api/status")
async def get_status(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(BotStatus).limit(1))
    status = result.scalar_one_or_none()
    account = {}
    try:
        account = alpaca_broker.get_account()
    except:
        pass
    return {
        "is_running": status.is_running if status else False,
        "kill_switch": getattr(risk_engine, "_kill_switch", False),
        "mode": alpaca_broker.mode,
        "message": status.message if status else "Stopped",
        "started_at": str(status.started_at) if status and status.started_at else None,
        "equity": account.get("equity", 0),
        "daily_pnl": risk_engine.daily_pnl,
        "open_positions": len(alpaca_broker.get_positions()) if account else 0,
    }


@app.post("/api/bot/start")
async def start_bot():
    if bot.running:
        return {"message": "Already running"}
    asyncio.create_task(bot.start())
    return {"message": "Bot started"}


@app.post("/api/bot/stop")
async def stop_bot():
    await bot.stop()
    return {"message": "Bot stopped"}


@app.post("/api/bot/kill-switch/on")
async def kill_on():
    risk_engine.activate_kill_switch()
    await alpaca_broker.cancel_all_orders()
    return {"message": "Kill switch activated"}


@app.post("/api/bot/kill-switch/off")
async def kill_off():
    risk_engine.deactivate_kill_switch()
    return {"message": "Kill switch deactivated"}


@app.post("/api/bot/emergency-close")
async def emergency_close():
    risk_engine.activate_kill_switch()
    await alpaca_broker.cancel_all_orders()
    await alpaca_broker.close_all_positions()
    return {"message": "EMERGENCY: All positions closed"}


class StartSelectedRequest(BaseModel):
    strategies: list


@app.post("/api/bot/start-selected")
async def start_selected(req: StartSelectedRequest):
    if bot.running:
        await bot.stop()
        await asyncio.sleep(1)
    asyncio.create_task(bot.start(strategy_names=req.strategies))
    return {"message": f"Started: {', '.join(req.strategies)}"}


@app.get("/api/strategies")
async def list_strategies():
    try:
        return {"strategies": get_strategy_info()}
    except Exception as e:
        return {"strategies": [], "error": str(e)}


@app.post("/api/strategies/reload")
async def reload_strategies():
    names = bot.reload_strategies()
    return {"message": f"Reloaded {len(names)} strategies", "strategies": names}


# ─────────────────────────────────────────────────────────────────
# CHART — original working code + volume + MACD + EMA added
# ─────────────────────────────────────────────────────────────────
@app.get("/api/chart4/{symbol}")
async def get_chart(symbol: str, days: int = 365, timeframe: str = "1Day"):
    import pandas as pd

    try:
        try:
            import ta as ta_lib

            has_ta = True
        except ImportError:
            has_ta = False

        tf_map = {
            "5min": "5Min",
            "15min": "15Min",
            "1hr": "1Hour",
            "4hr": "4Hour",
            "1D": "1Day",
            "1W": "1Week",
            "5Min": "5Min",
            "15Min": "15Min",
            "1Hour": "1Hour",
            "4Hour": "4Hour",
            "1Day": "1Day",
            "1Week": "1Week",
        }
        tf = tf_map.get(timeframe, "1Day")
        intraday = tf in ["5Min", "15Min", "1Hour", "4Hour"]
        if intraday:
            days_back = min(days, 59)
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            lim = 2000
        else:
            days_back = days * 7 if tf == "1Week" else days
            start = (datetime.now() - timedelta(days=days_back + 300)).strftime(
                "%Y-%m-%d"
            )
            lim = days + 300

        df = alpaca_broker.api.get_bars(symbol, tf, start=start, limit=lim).df
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        ts_col = next((c for c in df.columns if "time" in c), None)
        if ts_col:
            if intraday:
                df["ts"] = df[ts_col].astype(str).str[:16].str.replace("T", " ")
            else:
                df["ts"] = df[ts_col].astype(str).str[:10]
        df["close"] = pd.to_numeric(df["close"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
        n = len(df)

        def safe_ma(p):
            if n < p:
                return [None] * n
            return [
                round(v, 2) if v == v else None
                for v in df["close"].rolling(p).mean().tolist()
            ]

        def safe_ema(p):
            return [
                round(v, 2) if v == v else None
                for v in df["close"].ewm(span=p, adjust=False).mean().tolist()
            ]

        # RSI
        rsi_vals = [None] * n
        if n >= 14:
            try:
                if has_ta:
                    rsi_s = ta_lib.momentum.RSIIndicator(df["close"], window=14).rsi()
                    rsi_vals = [round(v, 2) if v == v else None for v in rsi_s.tolist()]
                else:
                    raise Exception("no ta")
            except:
                delta = df["close"].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, float("nan"))
                rsi_s = 100 - (100 / (1 + rs))
                rsi_vals = [round(v, 2) if v == v else None for v in rsi_s.tolist()]

        # MACD (always manual)
        macd_line = [None] * n
        macd_signal = [None] * n
        macd_hist = [None] * n
        if n >= 27:
            e12 = df["close"].ewm(span=12, adjust=False).mean()
            e26 = df["close"].ewm(span=26, adjust=False).mean()
            ml = e12 - e26
            ms = ml.ewm(span=9, adjust=False).mean()
            mh = ml - ms
            clean = lambda s: [round(v, 4) if v == v else None for v in s.tolist()]
            macd_line, macd_signal, macd_hist = clean(ml), clean(ms), clean(mh)

        # Supertrend (ATR period=10, multiplier=3.0)
        st_vals = [None] * n
        st_dir = [None] * n
        if n >= 12 and "high" in df.columns and "low" in df.columns:
            import numpy as np

            hl2 = (df["high"] + df["low"]) / 2
            tr_s = pd.concat(
                [
                    df["high"] - df["low"],
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1)),
                ],
                axis=1,
            ).max(axis=1)
            atr_s = tr_s.rolling(10).mean()
            ub = (hl2 + 3.0 * atr_s).tolist()
            lb = (hl2 - 3.0 * atr_s).tolist()
            cl_s = df["close"].tolist()
            direction = [1] * n
            for i in range(1, n):
                if (
                    lb[i] is None
                    or ub[i] is None
                    or lb[i - 1] is None
                    or ub[i - 1] is None
                ):
                    continue
                if cl_s[i] > (ub[i - 1] or 0):
                    direction[i] = 1
                elif cl_s[i] < (lb[i - 1] or 0):
                    direction[i] = -1
                else:
                    direction[i] = direction[i - 1]
                    if direction[i] == 1 and lb[i] < lb[i - 1]:
                        lb[i] = lb[i - 1]
                    if direction[i] == -1 and ub[i] > ub[i - 1]:
                        ub[i] = ub[i - 1]
                st_vals[i] = round(lb[i] if direction[i] == 1 else ub[i], 2)
                st_dir[i] = direction[i]

        # VWAP (rolling 20-bar proxy for daily VWAP)
        vwap_vals = [None] * n
        if "volume" in df.columns and n >= 5:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            vol = df["volume"].replace(0, 1)
            vwap_r = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
            vwap_vals = [round(v, 2) if v == v else None for v in vwap_r.tolist()]

        return {
            "dates": df["ts"].tolist(),
            "close": df["close"].round(2).tolist(),
            "open": df["open"].round(2).tolist(),
            "high": df["high"].round(2).tolist(),
            "low": df["low"].round(2).tolist(),
            "volume": df["volume"].round(0).tolist(),
            "ma21": safe_ma(21),
            "ma50": safe_ma(50),
            "ma200": safe_ma(200),
            "ema9": safe_ema(9),
            "ema21": safe_ema(21),
            "ema50": safe_ema(50),
            "rsi": rsi_vals,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "vwap": vwap_vals,
            "supertrend": st_vals,
            "st_dir": st_dir,
            "count": n,
        }
    except Exception as e:
        raise HTTPException(500, f"Chart error: {str(e)}")


# ─────────────────────────────────────────────────────────────────
# MULTI-CHART — 2x2 panel with different stocks/timeframes
# ─────────────────────────────────────────────────────────────────
@app.get("/api/multi-chart")
async def multi_chart(
    symbols: str = "AAPL,MSFT,NVDA,TSLA",
    timeframes: str = "1D,1D,1D,1D",
    days: int = 180,
):
    import pandas as pd

    sym_list = [x.strip().upper() for x in symbols.split(",")][:4]
    tf_list = [x.strip() for x in timeframes.split(",")]
    tf_map = {
        "5min": "5Min",
        "15min": "15Min",
        "1hr": "1Hour",
        "4hr": "4Hour",
        "1D": "1Day",
        "1W": "1Week",
    }
    results = []
    for i in range(min(4, len(sym_list))):
        sym = sym_list[i]
        tf = tf_map.get(tf_list[i] if i < len(tf_list) else "1D", "1Day")
        try:
            start = (datetime.now() - timedelta(days=days + 300)).strftime("%Y-%m-%d")
            df = alpaca_broker.api.get_bars(sym, tf, start=start, limit=days + 300).df
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            ts_col = next((c for c in df.columns if "time" in c), None)
            if ts_col:
                df["ts"] = df[ts_col].astype(str).str[:10]
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
            n = len(df)

            def sma(p):
                return (
                    [
                        round(v, 2) if v == v else None
                        for v in df["close"].rolling(p).mean().tolist()
                    ]
                    if n >= p
                    else [None] * n
                )

            # RSI manual
            rsi = [None] * n
            if n >= 15:
                d2 = df["close"].diff()
                g = d2.clip(lower=0).rolling(14).mean()
                l2 = (-d2.clip(upper=0)).rolling(14).mean()
                rs = g / l2.replace(0, float("nan"))
                r2 = 100 - (100 / (1 + rs))
                rsi = [round(v, 2) if v == v else None for v in r2.tolist()]
            # MACD manual
            ml = ms = mh = [None] * n
            if n >= 27:
                e12 = df["close"].ewm(span=12, adjust=False).mean()
                e26 = df["close"].ewm(span=26, adjust=False).mean()
                _ml = e12 - e26
                _ms = _ml.ewm(span=9, adjust=False).mean()
                _mh = _ml - _ms
                clean = lambda s: [round(v, 4) if v == v else None for v in s.tolist()]
                ml, ms, mh = clean(_ml), clean(_ms), clean(_mh)

            # EMA
            def ema_fn(p):
                return [
                    round(v, 2) if v == v else None
                    for v in df["close"].ewm(span=p, adjust=False).mean().tolist()
                ]

            # VWAP
            vwap_mc = [None] * n
            if "volume" in df.columns and n >= 5:
                tp_mc = (
                    (df["high"] + df["low"] + df["close"]) / 3
                    if "high" in df.columns
                    else df["close"]
                )
                vol_mc = df["volume"].replace(0, 1)
                vwap_mc = [
                    round(v, 2) if v == v else None
                    for v in (
                        (tp_mc * vol_mc).rolling(20).sum() / vol_mc.rolling(20).sum()
                    ).tolist()
                ]
            # Supertrend
            st_mc = [None] * n
            stdir_mc = [None] * n
            if n >= 12 and "high" in df.columns:
                hl2_mc = (df["high"] + df["low"]) / 2
                tr_mc = pd.concat(
                    [
                        df["high"] - df["low"],
                        abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"] - df["close"].shift(1)),
                    ],
                    axis=1,
                ).max(axis=1)
                atr_mc = tr_mc.rolling(10).mean()
                ub_mc = (hl2_mc + 3.0 * atr_mc).tolist()
                lb_mc = (hl2_mc - 3.0 * atr_mc).tolist()
                cl_mc = df["close"].tolist()
                dir_mc = [1] * n
                for i_mc in range(1, n):
                    if (
                        lb_mc[i_mc] is None
                        or ub_mc[i_mc] is None
                        or lb_mc[i_mc - 1] is None
                        or ub_mc[i_mc - 1] is None
                    ):
                        continue
                    if cl_mc[i_mc] > (ub_mc[i_mc - 1] or 0):
                        dir_mc[i_mc] = 1
                    elif cl_mc[i_mc] < (lb_mc[i_mc - 1] or 0):
                        dir_mc[i_mc] = -1
                    else:
                        dir_mc[i_mc] = dir_mc[i_mc - 1]
                        if dir_mc[i_mc] == 1 and lb_mc[i_mc] < lb_mc[i_mc - 1]:
                            lb_mc[i_mc] = lb_mc[i_mc - 1]
                        if dir_mc[i_mc] == -1 and ub_mc[i_mc] > ub_mc[i_mc - 1]:
                            ub_mc[i_mc] = ub_mc[i_mc - 1]
                    st_mc[i_mc] = round(
                        lb_mc[i_mc] if dir_mc[i_mc] == 1 else ub_mc[i_mc], 2
                    )
                    stdir_mc[i_mc] = dir_mc[i_mc]
            results.append(
                {
                    "symbol": sym,
                    "timeframe": tf_list[i] if i < len(tf_list) else "1D",
                    "dates": df["ts"].tolist(),
                    "close": df["close"].round(2).tolist(),
                    "open": df["open"].round(2).tolist()
                    if "open" in df.columns
                    else [],
                    "high": df["high"].round(2).tolist()
                    if "high" in df.columns
                    else [],
                    "low": df["low"].round(2).tolist() if "low" in df.columns else [],
                    "volume": df["volume"].round(0).tolist(),
                    "ma50": sma(50),
                    "ma200": sma(200),
                    "ema9": ema_fn(9),
                    "ema21": ema_fn(21),
                    "ema50": ema_fn(50),
                    "vwap": vwap_mc,
                    "supertrend": st_mc,
                    "st_dir": stdir_mc,
                    "rsi": rsi,
                    "macd": ml,
                    "macd_signal": ms,
                    "macd_hist": mh,
                }
            )
        except Exception as e:
            results.append(
                {"symbol": sym, "error": str(e), "dates": [], "close": [], "volume": []}
            )
    return {"charts": results}


# ─────────────────────────────────────────────────────────────────
# BACKTEST (original working code unchanged)
# ─────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol: str
    strategy: str = "both"
    days: int = 365
    cash: float = 10000
    fast_ma: int = 50
    slow_ma: int = 200
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70


@app.post("/api/backtest/run")
async def run_backtest(req: BacktestRequest):
    try:
        results = []
        if req.strategy in ("ma_crossover", "both"):
            bt = Backtester(req.symbol, req.days, req.cash)
            r = bt.run_ma_crossover(req.fast_ma, req.slow_ma)
            r["strategy"] = "MA Crossover"
            r["equity_curve"] = bt.equity_curve
            r["trades"] = bt.trades
            results.append(r)
        if req.strategy in ("rsi", "rsi_strategy", "both"):
            bt2 = Backtester(req.symbol, req.days, req.cash)
            r2 = bt2.run_rsi(req.rsi_period, req.oversold, req.overbought)
            r2["strategy"] = "RSI"
            r2["equity_curve"] = bt2.equity_curve
            r2["trades"] = bt2.trades
            results.append(r2)
        # New strategies via run method
        if req.strategy in ("macd_strategy", "macd"):
            try:
                bt3 = Backtester(req.symbol, req.days, req.cash)
                r3 = (
                    bt3.run_macd()
                    if hasattr(bt3, "run_macd")
                    else bt3.run_ma_crossover(req.fast_ma, req.slow_ma)
                )
                r3["strategy"] = "MACD"
                r3["equity_curve"] = bt3.equity_curve
                r3["trades"] = bt3.trades
                results.append(r3)
            except Exception as e:
                log.error(f"MACD bt error: {e}")
        if req.strategy in ("supertrend", "supertrend_strategy"):
            try:
                bt4 = Backtester(req.symbol, req.days, req.cash)
                r4 = (
                    bt4.run_supertrend()
                    if hasattr(bt4, "run_supertrend")
                    else bt4.run_ma_crossover(req.fast_ma, req.slow_ma)
                )
                r4["strategy"] = "Supertrend"
                r4["equity_curve"] = bt4.equity_curve
                r4["trades"] = bt4.trades
                results.append(r4)
            except Exception as e:
                log.error(f"Supertrend bt error: {e}")
        if not results:
            bt0 = Backtester(req.symbol, req.days, req.cash)
            r0 = bt0.run_ma_crossover(req.fast_ma, req.slow_ma)
            r0["strategy"] = req.strategy or "MA Crossover"
            r0["equity_curve"] = bt0.equity_curve
            r0["trades"] = bt0.trades
            results.append(r0)
        return {"status": "ok", "results": results}
    except Exception as e:
        raise HTTPException(500, f"Backtest error: {str(e)}")


class StrategyBacktestRequest(BaseModel):
    symbol: str
    strategy: str
    days: int = 365
    cash: float = 10000
    fast_ma: int = 50
    slow_ma: int = 200
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    risk_per_trade: float = 0.01


@app.post("/api/backtest/strategy")
async def backtest_strategy(req: StrategyBacktestRequest):
    try:
        import pandas as pd

        start = (datetime.now() - timedelta(days=req.days + 300)).strftime("%Y-%m-%d")
        df = alpaca_broker.api.get_bars(
            req.symbol, "1Day", start=start, limit=req.days + 300
        ).df
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        ts_col = next((c for c in df.columns if "time" in c), None)
        if ts_col:
            df["timestamp"] = df[ts_col].astype(str).str[:10]
        for col in ["close", "open", "high", "low"]:
            df[col] = pd.to_numeric(df[col])

        if req.strategy == "institutional_ema":
            df["ema9"] = df["close"].ewm(span=9).mean()
            df["ema21"] = df["close"].ewm(span=21).mean()
            df["ema50"] = df["close"].ewm(span=50).mean()
            df["ema200"] = df["close"].ewm(span=200).mean()
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["atr"] = df[["tr1", "tr2", "tr3"]].max(axis=1).rolling(14).mean()
            df = df.dropna().reset_index(drop=True)
            cash = req.cash
            shares = 0
            trades = []
            equity_curve = []
            entry_price = 0
            for i in range(1, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i - 1]
                price = row["close"]
                ts = row["timestamp"]
                bull = price > row["ema200"] and row["ema50"] > row["ema200"]
                pb = abs(price - row["ema21"]) / price < 0.015
                bcross = prev["ema9"] < prev["ema21"] and row["ema9"] > row["ema21"]
                vvol = row["atr"] / price > 0.01
                bear = prev["ema9"] > prev["ema21"] and row["ema9"] < row["ema21"]
                tbrk = price < row["ema200"]
                if bull and pb and bcross and vvol and shares == 0:
                    rps = 1.5 * row["atr"]
                    qty = int((cash * req.risk_per_trade) / rps) if rps > 0 else 0
                    qty = min(qty, int(cash * 0.95 / price))
                    if qty > 0:
                        cash -= qty * price
                        shares = qty
                        entry_price = price
                        trades.append(
                            {
                                "date": ts,
                                "side": "BUY",
                                "qty": qty,
                                "price": round(price, 2),
                                "reason": "EMA setup",
                                "value": qty * price,
                            }
                        )
                elif shares > 0 and (
                    bear or tbrk or price < entry_price - 1.5 * row["atr"]
                ):
                    pnl = (price - entry_price) * shares
                    cash += shares * price
                    trades.append(
                        {
                            "date": ts,
                            "side": "SELL",
                            "qty": shares,
                            "price": round(price, 2),
                            "reason": "Exit",
                            "pnl": round(pnl, 2),
                            "value": shares * price,
                        }
                    )
                    shares = 0
                equity_curve.append(
                    {"date": ts, "equity": round(cash + shares * price, 2)}
                )
            final = cash + shares * (df["close"].iloc[-1] if shares > 0 else 0)
            sells = [t for t in trades if t["side"] == "SELL"]
            wins = [t for t in sells if t.get("pnl", 0) > 0]
            return {
                "status": "ok",
                "results": [
                    {
                        "strategy": "Institutional EMA",
                        "symbol": req.symbol,
                        "starting_cash": req.cash,
                        "final_equity": round(final, 2),
                        "total_return_pct": round(
                            (final - req.cash) / req.cash * 100, 2
                        ),
                        "total_pnl": round(sum(t.get("pnl", 0) for t in sells), 2),
                        "total_trades": len(trades),
                        "winning_trades": len(wins),
                        "losing_trades": len(sells) - len(wins),
                        "win_rate_pct": round(len(wins) / len(sells) * 100, 2)
                        if sells
                        else 0,
                        "equity_curve": equity_curve,
                        "trades": trades,
                    }
                ],
            }
        else:
            bt = Backtester(req.symbol, req.days, req.cash)
            if req.strategy == "ma_crossover":
                r = bt.run_ma_crossover(req.fast_ma, req.slow_ma)
                r["strategy"] = "MA Crossover"
            else:
                r = bt.run_rsi(req.rsi_period, req.oversold, req.overbought)
                r["strategy"] = "RSI"
            r["equity_curve"] = bt.equity_curve
            r["trades"] = bt.trades
            return {"status": "ok", "results": [r]}
    except Exception as e:
        raise HTTPException(500, f"Strategy backtest error: {str(e)}")


@app.get("/api/health")
async def health():
    import os

    return {
        "status": "ok",
        "version": "3.0",
        "mode": alpaca_broker.mode,
        "alpaca_key_set": bool(
            os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        ),
    }


@app.get("/api/debug")
async def debug():
    import os, sys

    checks = {}
    try:
        acc = alpaca_broker.get_account()
        checks["alpaca"] = f"OK equity={acc['equity']}"
    except Exception as e:
        checks["alpaca"] = f"FAIL: {str(e)}"
    try:
        strats = get_strategy_info()
        checks["strategies"] = f"OK {len(strats)}: {[s['name'] for s in strats]}"
    except Exception as e:
        checks["strategies"] = f"FAIL: {str(e)}"
    checks["bot_running"] = bot.running
    checks["kill_switch"] = getattr(risk_engine, "_kill_switch", False)
    checks["python"] = sys.version
    checks["env_keys"] = {
        "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
        "APCA_API_KEY_ID": bool(os.getenv("APCA_API_KEY_ID")),
        "ALPACA_SECRET_KEY": bool(os.getenv("ALPACA_SECRET_KEY")),
        "APCA_API_SECRET_KEY": bool(os.getenv("APCA_API_SECRET_KEY")),
        "ALPACA_MODE": os.getenv("ALPACA_MODE", "not set"),
    }
    return checks


# ─────────────────────────────────────────────────────────────────
# ANALYTICS — portfolio position charts + scanner
# ─────────────────────────────────────────────────────────────────
@app.get("/api/portfolio-charts")
async def portfolio_charts(timeframe: str = "1Day", days: int = 90):
    """Return chart data for every open position (for Analytics tab)."""
    import pandas as pd

    tf_map = {
        "5min": "5Min", "15min": "15Min", "1hr": "1Hour",
        "4hr": "4Hour", "1D": "1Day", "1W": "1Week",
        "5Min": "5Min", "15Min": "15Min", "1Hour": "1Hour",
        "4Hour": "4Hour", "1Day": "1Day", "1Week": "1Week",
    }
    tf = tf_map.get(timeframe, "1Day")
    intraday = tf in ["5Min", "15Min", "1Hour", "4Hour"]

    try:
        positions = alpaca_broker.get_positions()
    except Exception as e:
        raise HTTPException(500, f"Could not fetch positions: {e}")

    results = []
    for pos in positions:
        sym = pos["symbol"]
        try:
            if intraday:
                days_back = min(days, 59)
                start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                lim = 2000
            else:
                start = (datetime.now() - timedelta(days=days + 300)).strftime("%Y-%m-%d")
                lim = days + 300

            df = alpaca_broker.api.get_bars(sym, tf, start=start, limit=lim).df
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            ts_col = next((c for c in df.columns if "time" in c), None)
            if ts_col:
                if intraday:
                    df["ts"] = df[ts_col].astype(str).str[:16].str.replace("T", " ")
                else:
                    df["ts"] = df[ts_col].astype(str).str[:10]
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
            n = len(df)

            # EMAs
            def ema(p):
                return [round(v, 2) if v == v else None
                        for v in df["close"].ewm(span=p, adjust=False).mean().tolist()]

            # RSI
            rsi_vals = [None] * n
            if n >= 15:
                d = df["close"].diff()
                g = d.clip(lower=0).rolling(14).mean()
                l = (-d.clip(upper=0)).rolling(14).mean()
                rs = g / l.replace(0, float("nan"))
                rsi_s = 100 - (100 / (1 + rs))
                rsi_vals = [round(v, 2) if v == v else None for v in rsi_s.tolist()]

            # MACD
            macd_line = macd_signal = macd_hist = [None] * n
            if n >= 27:
                e12 = df["close"].ewm(span=12, adjust=False).mean()
                e26 = df["close"].ewm(span=26, adjust=False).mean()
                ml = e12 - e26
                ms = ml.ewm(span=9, adjust=False).mean()
                mh = ml - ms
                clean = lambda s: [round(v, 4) if v == v else None for v in s.tolist()]
                macd_line, macd_signal, macd_hist = clean(ml), clean(ms), clean(mh)

            # VWAP
            vwap_vals = [None] * n
            if "volume" in df.columns and n >= 5:
                tp = (df["high"] + df["low"] + df["close"]) / 3
                vol = df["volume"].replace(0, 1)
                vwap_r = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
                vwap_vals = [round(v, 2) if v == v else None for v in vwap_r.tolist()]

            # Supertrend
            st_vals = [None] * n
            st_dir = [None] * n
            if n >= 12:
                hl2 = (df["high"] + df["low"]) / 2
                tr_s = pd.concat([
                    df["high"] - df["low"],
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1)),
                ], axis=1).max(axis=1)
                atr_s = tr_s.rolling(10).mean()
                ub = (hl2 + 3.0 * atr_s).tolist()
                lb = (hl2 - 3.0 * atr_s).tolist()
                cl_s = df["close"].tolist()
                direction = [1] * n
                for i in range(1, n):
                    if any(x is None for x in [lb[i], ub[i], lb[i-1], ub[i-1]]):
                        continue
                    if cl_s[i] > (ub[i-1] or 0):
                        direction[i] = 1
                    elif cl_s[i] < (lb[i-1] or 0):
                        direction[i] = -1
                    else:
                        direction[i] = direction[i-1]
                        if direction[i] == 1 and lb[i] < lb[i-1]:
                            lb[i] = lb[i-1]
                        if direction[i] == -1 and ub[i] > ub[i-1]:
                            ub[i] = ub[i-1]
                    st_vals[i] = round(lb[i] if direction[i] == 1 else ub[i], 2)
                    st_dir[i] = direction[i]

            results.append({
                "symbol": sym,
                "position": pos,
                "dates": df["ts"].tolist() if "ts" in df.columns else [],
                "close": df["close"].round(2).tolist(),
                "open": df["open"].round(2).tolist() if "open" in df.columns else [],
                "high": df["high"].round(2).tolist() if "high" in df.columns else [],
                "low": df["low"].round(2).tolist() if "low" in df.columns else [],
                "volume": df["volume"].round(0).tolist(),
                "ema9": ema(9), "ema21": ema(21), "ema50": ema(50),
                "rsi": rsi_vals,
                "macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
                "vwap": vwap_vals,
                "supertrend": st_vals, "st_dir": st_dir,
            })
        except Exception as e:
            results.append({"symbol": sym, "position": pos, "error": str(e),
                            "dates": [], "close": [], "volume": []})

    return {"positions": results, "count": len(results)}


@app.get("/api/scanner")
async def scanner(symbols: str = "AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,SPY,QQQ,AMD",
                  timeframe: str = "1Day", days: int = 60):
    """Technical scanner — returns signal summary for each symbol from Alpaca data."""
    import pandas as pd, numpy as np

    tf_map = {
        "5min": "5Min", "15min": "15Min", "1hr": "1Hour",
        "4hr": "4Hour", "1D": "1Day", "1W": "1Week",
    }
    tf = tf_map.get(timeframe, "1Day")
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]

    results = []
    for sym in sym_list:
        try:
            start = (datetime.now() - timedelta(days=days + 100)).strftime("%Y-%m-%d")
            df = alpaca_broker.api.get_bars(sym, tf, start=start, limit=days + 100).df
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
            n = len(df)
            if n < 20:
                results.append({"symbol": sym, "error": "Not enough data"})
                continue

            price = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2]) if n >= 2 else price
            chg_pct = (price - prev_close) / prev_close * 100

            ema9 = float(df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
            ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = float(df["close"].ewm(span=200, adjust=False).mean().iloc[-1]) if n >= 50 else None

            # RSI
            d = df["close"].diff()
            g = d.clip(lower=0).rolling(14).mean()
            lo = (-d.clip(upper=0)).rolling(14).mean()
            rs = g / lo.replace(0, float("nan"))
            rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if n >= 15 else None

            # MACD signal
            e12 = df["close"].ewm(span=12, adjust=False).mean()
            e26 = df["close"].ewm(span=26, adjust=False).mean()
            ml = e12 - e26
            ms_s = ml.ewm(span=9, adjust=False).mean()
            mh = ml - ms_s
            macd_hist_last = float(mh.iloc[-1]) if n >= 27 else None
            macd_hist_prev = float(mh.iloc[-2]) if n >= 28 else None

            # Volume
            avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
            last_vol = float(df["volume"].iloc[-1])
            vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

            # Supertrend direction
            hl2 = (df["high"] + df["low"]) / 2
            tr_s = pd.concat([
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ], axis=1).max(axis=1)
            atr_s = tr_s.rolling(10).mean()
            ub_arr = (hl2 + 3.0 * atr_s).tolist()
            lb_arr = (hl2 - 3.0 * atr_s).tolist()
            cl_arr = df["close"].tolist()
            direction = [1] * n
            for i in range(1, n):
                if any(x is None for x in [lb_arr[i], ub_arr[i], lb_arr[i-1], ub_arr[i-1]]):
                    continue
                if cl_arr[i] > (ub_arr[i-1] or 0): direction[i] = 1
                elif cl_arr[i] < (lb_arr[i-1] or 0): direction[i] = -1
                else:
                    direction[i] = direction[i-1]
                    if direction[i] == 1 and lb_arr[i] < lb_arr[i-1]: lb_arr[i] = lb_arr[i-1]
                    if direction[i] == -1 and ub_arr[i] > ub_arr[i-1]: ub_arr[i] = ub_arr[i-1]
            st_direction = direction[-1]

            # Signal scoring
            signals = []
            score = 0
            if price > ema9 > ema21: signals.append("EMA Bull"); score += 1
            elif price < ema9 < ema21: signals.append("EMA Bear"); score -= 1
            if ema9 > ema21 > ema50: signals.append("Trend Up"); score += 1
            elif ema9 < ema21 < ema50: signals.append("Trend Down"); score -= 1
            if rsi and rsi < 35: signals.append("RSI OS"); score += 1
            elif rsi and rsi > 65: signals.append("RSI OB"); score -= 1
            if macd_hist_last and macd_hist_prev:
                if macd_hist_last > 0 and macd_hist_prev <= 0: signals.append("MACD Cross↑"); score += 2
                elif macd_hist_last < 0 and macd_hist_prev >= 0: signals.append("MACD Cross↓"); score -= 2
                elif macd_hist_last > 0: signals.append("MACD+"); score += 0.5
            if st_direction == 1: signals.append("ST Bull"); score += 1
            elif st_direction == -1: signals.append("ST Bear"); score -= 1
            if vol_ratio > 1.5: signals.append(f"Vol×{vol_ratio:.1f}"); score += 0.5

            if score >= 3: overall = "STRONG BUY"
            elif score >= 1: overall = "BUY"
            elif score <= -3: overall = "STRONG SELL"
            elif score <= -1: overall = "SELL"
            else: overall = "NEUTRAL"

            results.append({
                "symbol": sym,
                "price": round(price, 2),
                "chg_pct": round(chg_pct, 2),
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "ema50": round(ema50, 2),
                "ema200": round(ema200, 2) if ema200 else None,
                "rsi": round(rsi, 1) if rsi else None,
                "macd_hist": round(macd_hist_last, 4) if macd_hist_last else None,
                "vol_ratio": round(vol_ratio, 2),
                "st_dir": st_direction,
                "score": round(score, 1),
                "signals": signals,
                "overall": overall,
            })
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    return {"results": results, "count": len(results)}


@app.get("/api/pnl-summary")
async def pnl_summary(db: AsyncSession = Depends(get_db)):
    """Detailed P&L breakdown: realized + unrealized."""
    import json

    # Realized from DB
    result = await db.execute(select(Trade).order_by(desc(Trade.created_at)))
    trades = result.scalars().all()
    realized = sum(t.pnl or 0 for t in trades if t.side == "sell" and t.pnl is not None)

    # Unrealized from Alpaca
    unrealized = 0.0
    positions = []
    try:
        pos_raw = alpaca_broker.get_positions()
        for p in pos_raw:
            unrealized += p["unrealized_pnl"]
        positions = pos_raw
    except:
        pass

    account = {}
    try:
        account = alpaca_broker.get_account()
    except:
        pass

    # Daily grouped realized
    from collections import defaultdict
    daily = defaultdict(float)
    for t in trades:
        if t.pnl and t.side == "sell":
            day = str(t.created_at)[:10]
            daily[day] += t.pnl

    return {
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(realized + unrealized, 2),
        "equity": account.get("equity", 0),
        "cash": account.get("cash", 0),
        "buying_power": account.get("buying_power", 0),
        "positions": positions,
        "daily_pnl": [{"date": k, "pnl": round(v, 2)} for k, v in sorted(daily.items())],
        "total_trades": len(trades),
        "winning_trades": sum(1 for t in trades if (t.pnl or 0) > 0 and t.side == "sell"),
        "losing_trades": sum(1 for t in trades if (t.pnl or 0) < 0 and t.side == "sell"),
    }
