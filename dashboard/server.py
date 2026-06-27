import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from core.config import DASHBOARD_PASSWORD
from core.database import (
    init_db, get_recent_trades, get_recent_signals,
    is_kill_switch_on, set_kill_switch, get_setting, set_setting
)
from brokers.alpaca_adapter import AlpacaAdapter
from webhook.handler import process_webhook
from core.logger import get_logger

log = get_logger("server")
app = FastAPI(title="OptiTrade Webhook Bot")
security = HTTPBasic()
alpaca = AlpacaAdapter()

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    log.info("Server started ✅")

# ── Auth helper ───────────────────────────────────────────────────────────────

def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not correct:
        raise HTTPException(status_code=401, detail="Incorrect password",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ── Webhook endpoint (no auth — TradingView posts here) ──────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    result = process_webhook(payload)
    return JSONResponse(content=result)

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "kill_switch": is_kill_switch_on()}

# ── Dashboard API ─────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def api_portfolio(user=Depends(require_auth)):
    try:
        return alpaca.get_portfolio_summary()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/trades")
async def api_trades(user=Depends(require_auth)):
    return get_recent_trades(50)

@app.get("/api/signals")
async def api_signals(user=Depends(require_auth)):
    return get_recent_signals(50)

@app.get("/api/status")
async def api_status(user=Depends(require_auth)):
    return {
        "kill_switch":      is_kill_switch_on(),
        "trading_enabled":  get_setting("trading_enabled") == "1",
    }

@app.post("/api/kill-switch")
async def api_kill_switch(request: Request, user=Depends(require_auth)):
    body = await request.json()
    on = body.get("on", True)
    set_kill_switch(on)
    return {"kill_switch": on}

@app.post("/api/trading-enabled")
async def api_trading_enabled(request: Request, user=Depends(require_auth)):
    body = await request.json()
    enabled = body.get("enabled", True)
    set_setting("trading_enabled", "1" if enabled else "0")
    return {"trading_enabled": enabled}

@app.post("/api/close-all")
async def api_close_all(user=Depends(require_auth)):
    try:
        alpaca.close_all_positions()
        return {"status": "ok", "message": "All positions closed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ── Dashboard UI ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(user=Depends(require_auth)):
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()
