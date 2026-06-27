# PROJECT REFERENCE — OptiTrade AI → Alpaca Trading Bot v8

*Share this file with Claude in any future session to resume exactly where we left off.*

---

## What This Does

Full-stack automated trading bot:
- Receives BUY/SELL signals from TradingView (OptiTrade AI) via webhooks
- Places orders automatically on Alpaca paper/live account
- **9-tab** web dashboard: Dashboard, Positions, History, Analyzer Pro, Research, Sessions, EOD Journal, Analytics, MirrorFish AI, Settings
- Telegram alerts on every trade + bot commands (/positions /pnl /close AAPL etc)
- Analyzer Pro: real-time RSI/MACD/ADX/BB/EMA50/VWAP scanner (pure Python, yfinance)
- Research engine: 13F institutional, earnings whiplash, sector rotation, insider+options
- **Analytics tab (NEW v8)**: per-symbol win rates, P&L charts, equity curve, profit factor
- **MirrorFish AI (NEW v8)**: free LLM market predictions via Groq/OpenRouter/HuggingFace
- Excel export of all trades and P&L

---

## GitHub

https://github.com/ambrose2107/TV_GPT_ALP

---

## Tech Stack

| Layer         | Tech                                                            |
| ------------- | --------------------------------------------------------------- |
| Language      | Python 3.11                                                     |
| Framework     | Flask 3.0                                                       |
| Database      | SQLite (trades.db)                                              |
| Broker        | Alpaca Markets API                                              |
| Market Data   | Yahoo Finance (via yfinance + direct API) — FREE, no key needed |
| SEC Data      | SEC EDGAR public API — FREE                                     |
| Hosting       | Railway (24/7)                                                  |
| Dev           | Replit                                                          |
| Signals       | TradingView + OptiTrade AI indicator                            |
| Notifications | Telegram Bot API                                                |
| Export        | openpyxl (Excel)                                                |
| Charts        | Chart.js (CDN, no install)                                      |
| AI Layer      | Groq / OpenRouter / HuggingFace (all free tiers, uses `requests`) |

---

## Environment Variables (Railway → Variables)

```
ALPACA_API_KEY        Alpaca API key
ALPACA_SECRET_KEY     Alpaca secret key
ALPACA_MODE           paper (or live when ready)
APP_SECRET_KEY        Any random string (Flask session encryption)
DASHBOARD_PASSWORD    Password for dashboard login
WEBHOOK_SECRET        Must match "secret" field in TradingView JSON
TELEGRAM_BOT_TOKEN    From @BotFather on Telegram
TELEGRAM_CHAT_ID      From @userinfobot on Telegram
MAX_POSITION_SIZE     10 (max shares per order)
DAILY_LOSS_LIMIT      500 (USD)
MAX_OPEN_POSITIONS    5
KILL_SWITCH           false

# v8 NEW — MirrorFish AI (add at least one; all are free tier)
GROQ_API_KEY          Get free at console.groq.com  ← RECOMMENDED (fastest, LLaMA 70B)
OPENROUTER_API_KEY    Get free at openrouter.ai     ← good fallback
HUGGINGFACE_API_KEY   Get free at huggingface.co    ← slowest but always works
```

---

## File Structure

```
main.py                           Entry point (gunicorn: main:app)
app.py                            v8: registers mirrorfish_bp + analytics_bp
requirements.txt                  No new packages — uses requests already in list
Procfile                          Railway deploy command
railway.json                      Railway config
.env.example                      Copy to .env for local dev

core/
  config.py                       All settings from env vars
  database.py                     v8: dual-timezone in get_recent_webhooks() + get_closed_positions()
  logger.py                       Centralised logging
  telegram.py                     Telegram alerts + bot message sender
  excel_export.py                 3-sheet Excel export
  analyzer.py                     Analyzer Pro — RSI/MACD/ADX/BB/EMA50/VWAP
  market_data.py                  Free data: Alpaca → Yahoo → Demo
  data_engine.py                  yfinance + SEC EDGAR
  timezone_utils.py               UTC/JST/NY with DST
  order_sync.py                   Sync Alpaca filled orders to local DB
  risk_engine.py                  Risk checks
  analytics_routes.py             NEW v8: /api/analytics/* — per-symbol stats

brokers/
  alpaca_adapter.py               Alpaca REST API

webhook/
  handler.py                      Signal processor
  routes.py                       POST /webhook  GET /health

mirrorfish/                       NEW v8
  __init__.py
  engine.py                       Multi-LLM: Groq → OpenRouter → HuggingFace
  routes.py                       /api/mirrorfish/status|analyze|portfolio|chat

dashboard/
  routes.py                       All API routes
  templates/
    login.html                    Login page
    dashboard.html                9-tab dashboard (v8)

research/
  sec_filings.py                  SEC EDGAR 13F institutional
  earnings.py                     Earnings whiplash scanner
  sector_rotation.py              Sector rotation detector
  insider_flow.py                 Form 4 insider buys + options flow
  ai_research.py                  AI research helpers
```

---

## Dashboard Tabs (v8)

| Tab            | Features                                                           |
| -------------- | ------------------------------------------------------------------ |
| Dashboard      | Account stats, kill switch, webhook URL, signals log (dual-tz)    |
| Positions      | Open positions with P&L, totals row, close button                 |
| History        | P&L summary, closed positions (dual-tz), all trades, Excel        |
| Analyzer Pro   | Signal grid, price charts, EMA overlay, compare tab               |
| Research       | 13F institutional, earnings whiplash, sector rotation, insider     |
| Sessions       | ICT session analysis: Asia / London / New York                     |
| EOD Journal    | P&L calendar, equity curve, Excel export                           |
| Analytics ★NEW | Per-symbol trade count, P&L bars, win rate bars, equity curve      |
| MirrorFish ★NEW| Symbol AI analysis, portfolio health, free-form chat               |
| Settings       | Env var reference, Telegram setup, TradingView JSON template       |

---

## v8 Changes

### FIX 1: Recent Signals now shows dual timezone (US + JST)
- `core/database.py`: `get_recent_webhooks()` adds `timestamp_display` field
  - Format: `"2026-05-23 09:35 EDT  |  22:35 JST"` via inline pytz formatting
- `dashboard/templates/dashboard.html`: webhook log uses `w.timestamp_display`

### FIX 2: equityChart initialization error
- All chart canvases now call `Chart.getChart(ctx)` before creating new chart
- Existing pattern in v7 extended to new Analytics charts

### NEW: Analytics Tab (tab 7)
- `core/analytics_routes.py`: Flask Blueprint `/api/analytics/summary`
- Computes per-symbol: trades, wins, losses, win rate %, total P&L, avg win/loss, profit factor, best/worst
- 4 Chart.js charts: equity curve, stacked wins/losses bar, P&L bar, win rate bar
- Full per-symbol breakdown table with colour coding

### NEW: MirrorFish AI Tab (tab 8)
- `mirrorfish/engine.py`: multi-provider LLM client (Groq → OpenRouter → HuggingFace)
  - Groq is free at console.groq.com — LLaMA 3.3 70B, fastest
  - Falls back automatically to next available provider
  - Zero new pip packages needed (uses `requests` already installed)
- `mirrorfish/routes.py`: 4 endpoints
  - `GET  /api/mirrorfish/status`    — which providers are configured
  - `POST /api/mirrorfish/analyze`   — AI analysis of a symbol
  - `POST /api/mirrorfish/portfolio` — AI commentary on full portfolio
  - `POST /api/mirrorfish/chat`      — free-form market chat
- Tab features: symbol analyzer card, portfolio health card, chat box with quick prompts
- Built-in setup guide shown when no API key configured

### Unchanged from v7
- All v7 bugfixes (chart date ranges, order sync, validate_symbol fail-open)
- All existing tabs: Dashboard, Positions, History, Analyzer Pro, Research, Sessions, EOD Journal, Settings
- Telegram bot commands, kill switch, Excel export
- All 29 tests still pass

---

## Getting Free API Keys for MirrorFish

| Provider     | URL                   | Time   | Model               | Var Name              |
| ------------ | --------------------- | ------ | ------------------- | --------------------- |
| Groq         | console.groq.com      | 2 min  | LLaMA 3.3 70B       | GROQ_API_KEY          |
| OpenRouter   | openrouter.ai         | 2 min  | LLaMA 3.3 70B :free | OPENROUTER_API_KEY    |
| HuggingFace  | huggingface.co/settings | 1 min | Mistral 7B          | HUGGINGFACE_API_KEY   |

All run online on Railway — no local GPU or laptop needed.

---

## Railway Deploy

1. Push this entire folder to GitHub
2. Railway auto-deploys from main branch
3. Add any new env vars (GROQ_API_KEY etc) in Railway → Variables
4. Visit `/api/mirrorfish/status` to verify AI provider is detected
5. Visit `/api/analytics/summary` to verify analytics route is live

---

## Tests

Run: `python test_bot.py`
All 29 existing tests pass. v8 adds no breaking changes to existing modules.
