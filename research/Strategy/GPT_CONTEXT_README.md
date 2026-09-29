# GPT Strategy Research Context — Living Project README

> **Purpose:** This file is the hand-off/context document for future GPT/AI sessions working on this repository.
> If a new chat is pointed to this file, it should understand what we are building, what has already been changed, and how future work should be done.

## 1. Current project

Repository: `ambrose2107/TV_GPT_ALP_V2`
Active development branch: `GPTver_confluence_v4`

This project is an automated TradingView → Flask/Railway → Alpaca trading system, with a growing **Strategy Lab / research layer** used to backtest, compare, export, and optimize strategies.

The current focus is **data-driven strategy research and optimization**, especially SPY and other strategy research runs, while keeping the production trading architecture stable.

## 2. Current development principles

- Keep the strategy architecture simple and maintainable.
- Prefer evidence from actual trade data over adding more filters.
- Do not optimize only for a higher backtest number.
- Look for robustness and avoid overfitting.
- Separate **backtest/Strategy Lab trades** from **actual Alpaca execution trades**.
- Preserve complete raw data in exports so a future AI session can independently analyze it.
- When improving a strategy, first identify *why* losses happen, then change the smallest necessary part of the strategy.
- Track P&L as a function of time, not only aggregate statistics.

## 3. Strategy Lab / research export standard

The Strategy Lab has export functionality intended specifically for AI/GitHub analysis.

### AI/GitHub JSON export — V2

Current schema:

`strategy-lab-v2-readable-2`

The JSON should contain:

### Strategy results
- Full metrics returned by every completed Strategy Lab run.
- Strategy parameters/configuration where available.
- Individual closed/backtest trades.
- Symbol, timeframe, data source and run information.

### Actual Alpaca data
The export now also includes an `alpaca` section containing:
- `provider`
- `synced_days`
- `sync_error`
- `closed_metrics`
- `order_log` — locally stored Alpaca order/execution records.
- `closed_positions` — locally stored closed Alpaca positions including P&L fields.

This distinction is intentional:

`strategies.*` = research/backtest results

`alpaca.*` = actual broker/execution history stored by the application

Never silently mix these two datasets when analyzing performance.

## 4. Alpaca trade synchronization

The dashboard already synchronizes recent Alpaca orders through `core.order_sync`.

The Strategy Lab now exposes:

`/api/strategy-lab/alpaca-trades`

This endpoint:
1. Attempts a 30-day Alpaca order sync.
2. Returns the locally stored order log.
3. Returns all stored closed positions.
4. Returns aggregate closed-position metrics.
5. Reports sync errors instead of hiding them.

The database contains:
- `trades` — order/trade log with Alpaca IDs.
- `closed_positions` — completed positions with entry price, exit price, P&L, P&L %, side, hold time and Alpaca ID.

Relevant functions include:
- `get_all_trades()`
- `get_all_closed_positions()`
- `get_closed_summary()`

## 5. Recent implemented updates

### P&L trend with time
The Strategy Lab was updated to support **P&L trend versus time** for the research strategies.

The objective is to identify:
- equity/P&L progression,
- periods of deterioration,
- clustering of losses,
- changes across time,
- whether performance is dependent on a particular market regime.

### AI-readable JSON V2
Added a structured UTF-8 JSON export designed to be directly readable from GitHub without requiring the Excel file.

It contains:
- schema version,
- export version,
- timestamp,
- symbol,
- strategy results,
- full metrics,
- individual strategy trades,
- actual Alpaca order history,
- actual Alpaca closed trades,
- Alpaca aggregate metrics.

### Excel export
The Strategy Lab Excel export supports:
- README/metadata,
- requested Alpaca OHLCV timeframes,
- strategy trade rows.

The Excel is useful for human analysis, but **JSON is the preferred hand-off format for GPT/AI analysis** because it is text-readable directly from GitHub.

## 6. Important files

### Strategy research
- `research/Strategy/`
- `research/Strategy/Main`

### Strategy Lab routes/export
- `research/xauusd_research_v2_routes.py`

Important endpoints/functions include:
- Strategy Lab export
- Strategy Lab Excel export
- Strategy Lab Alpaca trade export

### Dashboard
- `dashboard/routes.py`
- `dashboard/templates/xauusd_strategy_lab.html`

### Database
- `core/database.py`

### Broker
- `brokers/alpaca_adapter.py`

### Order synchronization
- `core/order_sync.py`

## 7. Current commits / milestones

Recent branch history includes:

- `da628132` — Show PnL trend with time for all V2 strategies
- `c036538` — Add full Alpaca trade export endpoint for Strategy Lab JSON
- `3bd6fd0` — Include Alpaca trades and full metrics in AI JSON export

Branch:
`GPTver_confluence_v4`

## 8. Current SPY research workflow

A research Excel has been generated/used under:

`research/Strategy/SPY_strategy_research (1).xlsx`

The important goal is to analyze the trade data and then map evidence back into the strategy code.

Analysis should cover at minimum:
- total trades,
- wins/losses,
- win rate,
- profit factor,
- expectancy,
- total P&L,
- max drawdown,
- R distribution,
- long vs short,
- time/session,
- trend vs range,
- volatility regime,
- entry type,
- stop-loss/take-profit behavior,
- loss clustering,
- P&L over time.

Then determine which code conditions actually contribute to losses or robustness.

Do **not** add filters simply because they improve one backtest. Prefer changes supported by repeated patterns and out-of-sample/robustness checks.

## 9. How future GPT sessions should work

When a new GPT session is pointed at this README:

1. Read this file first.
2. Inspect the current branch and latest commits before proposing changes.
3. Read the current Strategy Lab export format.
4. Use the latest JSON export whenever possible instead of asking the user to upload Excel.
5. Treat Strategy Lab/backtest trades and Alpaca live/paper trades as separate datasets.
6. Analyze the actual trade distribution before changing strategy logic.
7. Keep changes modular and minimal.
8. Add/update tests where practical.
9. Commit changes to `GPTver_confluence_v4` unless the user specifies another branch.
10. Update this README whenever a meaningful architecture, export, strategy, or research-workflow change is made.

## 10. What we are trying to achieve

The long-term objective is a **simple, scalable, data-driven trading research and execution system**.

The research loop is:

`Run strategy → Export complete data → Analyze trades → Identify real failure patterns → Make minimal code change → Backtest → Robustness check → Compare → Deploy`

The AI should optimize for **robustness and sustainable performance**, not merely the highest historical backtest result.

## 11. Do not lose these distinctions

### Backtest vs live/paper
A strategy can look profitable in backtesting while actual Alpaca execution behaves differently because of:
- fills,
- slippage,
- timing,
- position sizing,
- execution rules,
- market conditions.

Therefore both datasets must remain available.

### Metrics vs individual trades
Aggregate metrics explain *what happened*.
Individual trades explain *why it happened*.

Future analysis should use both.

### More rules vs better rules
Do not assume more filters = better strategy.
First establish whether a rule addresses a repeatable failure mode.

---

**Last updated:** 2026-09-29
**Context owner:** GPT-assisted strategy research workflow
