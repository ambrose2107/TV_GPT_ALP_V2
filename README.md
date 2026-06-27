# OptiTrade AI → Alpaca Trading Bot

Automated trading bot that receives signals from **TradingView (OptiTrade AI indicator)**
via webhooks and places orders on **Alpaca** (paper or live).

Hosted on **Railway**, coded on **Replit**.

---

## How it works

```
TradingView (OptiTrade signal fires)
        ↓  POST JSON to /webhook
Railway Server (this app, always running)
        ↓  validates secret, parses action
Alpaca Paper/Live Account
        ↓  places BUY / SELL order
Dashboard  (monitor from browser)
```

---

## Quick Setup

### 1. Alpaca Paper Account
- Go to https://alpaca.markets → sign up free
- Dashboard → API Keys → Generate Paper Trading keys
- Copy `API Key` and `Secret Key`

### 2. Environment Variables (Replit Secrets / Railway Variables)
```
ALPACA_API_KEY      = your_alpaca_api_key
ALPACA_SECRET_KEY   = your_alpaca_secret_key
ALPACA_MODE         = paper
APP_SECRET_KEY      = any_random_string_here
DASHBOARD_PASSWORD  = your_chosen_password
WEBHOOK_SECRET      = my_secret_123       ← must match TradingView JSON
```

### 3. Install & Run locally
```bash
pip install -r requirements.txt
python main.py
# Opens on http://localhost:8000
```

### 4. Run tests
```bash
python test_bot.py
# Should show: ✅ ALL TESTS PASSED (23 tests)
```

---

## TradingView Setup

### Step 1 — Open OptiTrade 2.0 indicator settings
Go to the **Inputs** tab. You'll see:
- **Long Entry Custom Alert Message**
- **Short Entry Custom Alert Message**

### Step 2 — Paste these JSONs

**Long Entry (BUY) box:**
```json
{
  "secret": "my_secret_123",
  "symbol": "AAPL",
  "action": "buy",
  "quantity": 1,
  "order_type": "market"
}
```

**Short Entry (SELL) box:**
```json
{
  "secret": "my_secret_123",
  "symbol": "AAPL",
  "action": "sell",
  "quantity": 1,
  "order_type": "market"
}
```

### Step 3 — Create the TradingView Alert
1. Click the **Alert** button (clock icon) on TradingView
2. Condition: select **OptiTrade 2.0**
3. Under **Notifications**: enable **Webhook URL**
4. Paste: `https://YOUR-RAILWAY-URL/webhook`
5. Message box: `{{strategy.order.alert_message}}`
6. Click **Create**

---

## Railway Hosting (New Account)

1. Go to https://railway.app → sign up with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Connect your GitHub and select this repo
4. Go to project **Settings → Variables** and add all env vars above
5. Railway auto-deploys. Copy your URL from **Settings → Domains**
6. Your webhook URL = `https://YOUR-DOMAIN.railway.app/webhook`

---

## Dashboard

Visit `https://YOUR-RAILWAY-URL/` in browser.
Login with your `DASHBOARD_PASSWORD`.

Features:
- Live account balance & buying power
- Today's P&L
- Open positions with unrealized P&L
- Trade log (last 50 trades)
- Webhook log (last 20 signals received)
- 🚨 Close All Positions button
- 🔴 Kill Switch (halts all new trades instantly)
- Auto-refreshes every 30 seconds

---

## Webhook Payload Reference

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `secret` | ✅ | string | Must match `WEBHOOK_SECRET` env var |
| `symbol` | ✅ | e.g. `"AAPL"` | Stock ticker (uppercase) |
| `action` | ✅ | `buy` / `sell` / `close` / `close_all` | Trade action |
| `quantity` | ✅ | number | Number of shares |
| `order_type` | optional | `market` (default) / `limit` | Order type |
| `price` | optional | number | Required only if `order_type = limit` |

---

## Risk Controls (set via env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_POSITION_SIZE` | 10 | Max shares per single order |
| `DAILY_LOSS_LIMIT` | 500 | USD — auto-stops if exceeded |
| `MAX_OPEN_POSITIONS` | 5 | Max simultaneous positions |
| `KILL_SWITCH` | false | Set `true` to halt all trading |

---

## Project Structure

```
trading-bot/
├── main.py                    # Entry point (gunicorn imports app from here)
├── app.py                     # Flask app factory
├── test_bot.py                # Full test suite (23 tests)
├── requirements.txt
├── Procfile                   # Railway/Heroku deploy command
├── railway.json               # Railway config
├── .env.example               # Copy to .env for local dev
├── core/
│   ├── config.py              # All settings from env vars
│   ├── database.py            # SQLite — trade log + webhook log
│   └── logger.py              # Centralised logging
├── brokers/
│   └── alpaca_adapter.py      # Alpaca API wrapper (buy/sell/close/positions)
├── webhook/
│   ├── handler.py             # Core signal processing logic
│   └── routes.py              # Flask /webhook and /health endpoints
└── dashboard/
    ├── routes.py              # Flask dashboard API + pages
    └── templates/
        ├── login.html         # Password login page
        └── dashboard.html     # Live monitoring dashboard
```

---

## Supported Indicators

| Indicator | Strategy Type | Alert Boxes | Notes |
|-----------|--------------|-------------|-------|
| OptiTrade 2.0 Buy-Sell | Flip (BUY flips to SELL) | Long Entry / Short Entry | ✅ Recommended for automation |
| OptiTrade 2.0 HWR | Trend with separate exits | Long / Short | Needs 4 alert messages |
| OptiTrade 2.0 TP-SL | Entry + TP1-4 + SL | Multiple | Complex — manual TP/SL recommended |

---

## Adding More Stocks

Just change the `symbol` field in the TradingView JSON, or create separate alerts per stock.
For multiple stocks, create one alert per stock on TradingView, each with the correct symbol in the JSON.
