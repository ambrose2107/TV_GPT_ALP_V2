"""Alpaca account transaction history for the Analytics dashboard."""
import os
from collections import defaultdict
from datetime import datetime
import requests
from flask import Blueprint, jsonify, request, session
from core.config import Config
from core.logger import get_logger

logger = get_logger(__name__)
account_history_bp = Blueprint("account_history", __name__)
PAGE_SIZE, MAX_PAGES = 100, 10
DIVIDEND_TYPES = {"DIV", "DIVCGL", "DIVCGS", "DIVROC", "DIVTXEX", "CGD"}
WITHHOLDING_TYPES = {"DIVNRA", "DIVFT", "DIVTW", "DIVPWH", "INTNRA", "INTTW", "TAX", "TAXW"}
FEE_TYPES = {"FEE", "DIVFEE", "CFEE", "PTC"}
ACTIVITY_LABELS = {
    "FILL":"Trade fill","CSD":"Deposit","CSW":"Withdrawal","DIV":"Dividend",
    "DIVCGL":"Capital-gain dividend (long)","DIVCGS":"Capital-gain dividend (short)",
    "DIVROC":"Return of capital","DIVTXEX":"Tax-exempt dividend","DIVNRA":"Non-resident tax withheld",
    "DIVFT":"Foreign tax withheld","DIVTW":"TEFRA tax withheld","DIVPWH":"Dividend product withholding",
    "INT":"Interest","INTNRA":"Interest tax withheld","INTTW":"Interest TEFRA withheld",
    "FEE":"Regulatory / trading fee","DIVFEE":"Dividend fee","CFEE":"Crypto fee","PTC":"Pass-through charge",
    "ACATC":"ACATS cash transfer","ACATS":"ACATS securities transfer","JNLC":"Cash journal",
    "JNLS":"Securities journal","MA":"Merger / acquisition","REORG":"Reorganization","SPIN":"Stock spinoff",
    "SPLIT":"Stock split"
}

def _number(value):
    try:
        result = float(value or 0)
        return result if result == result and abs(result) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0

def _headers():
    key = Config.ALPACA_API_KEY or os.environ.get("APCA_API_KEY_ID", "") or os.environ.get("ALPACA_KEY", "")
    secret = Config.ALPACA_SECRET_KEY or os.environ.get("APCA_API_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET", "")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def _date_of(activity):
    return str(activity.get("transaction_time") or activity.get("date") or activity.get("created_at") or "")[:10]

def _normalise_activity(activity):
    code = str(activity.get("activity_type") or activity.get("type") or "OTHER").upper()
    qty, price = _number(activity.get("qty")), _number(activity.get("price"))
    raw_amount, side = _number(activity.get("net_amount")), str(activity.get("side") or "").lower()
    if code == "FILL":
        gross = qty * price
        cash_flow = -gross if side in ("buy", "buy_minus") else gross
    else:
        cash_flow = raw_amount
    return {
        "id":str(activity.get("id") or ""), "date":_date_of(activity), "activity_type":code,
        "label":ACTIVITY_LABELS.get(code, code.replace("_"," ").title()),
        "symbol":str(activity.get("symbol") or ""), "description":str(activity.get("description") or activity.get("comment") or ""),
        "side":side, "qty":qty, "price":price, "amount":round(cash_flow,2),
        "net_amount":round(raw_amount,2), "order_id":str(activity.get("order_id") or ""),
        "currency":str(activity.get("currency") or "USD")
    }

def _realised_fifo(fills):
    """Long-only FIFO realised P&L; unmatched sells are excluded, not guessed."""
    queues, realised, orphan_qty = defaultdict(list), 0.0, 0.0
    ordered = sorted(fills, key=lambda x:(x.get("transaction_time") or x.get("date") or ""))
    for item in ordered:
        symbol, side = str(item.get("symbol") or ""), str(item.get("side") or "").lower()
        qty, price = _number(item.get("qty")), _number(item.get("price"))
        if not symbol or qty <= 0 or price <= 0: continue
        if side in ("buy","buy_minus"):
            queues[symbol].append([qty,price])
        elif side in ("sell","sell_plus"):
            remaining = qty
            while remaining > 1e-10 and queues[symbol]:
                lot=queues[symbol][0]; matched=min(remaining,lot[0])
                realised += (price-lot[1])*matched
                lot[0]-=matched; remaining-=matched
                if lot[0] <= 1e-10: queues[symbol].pop(0)
            orphan_qty += max(0.0,remaining)
    return round(realised,2), round(orphan_qty,8)

def build_history_summary(activities, positions, account):
    """Pure calculation helper for deterministic tests."""
    totals, fills, rows = defaultdict(float), [], []
    for raw in activities:
        code = str(raw.get("activity_type") or raw.get("type") or "OTHER").upper()
        item = _normalise_activity(raw); rows.append(item)
        amount = _number(raw.get("net_amount"))
        if code == "FILL":
            fills.append(raw); gross=item["qty"]*item["price"]
            if item["side"] in ("buy","buy_minus"): totals["gross_buys"] += gross
            elif item["side"] in ("sell","sell_plus","sell_short","sell_short_exempt"): totals["gross_sells"] += gross
        elif code == "CSD": totals["deposits"] += max(0.0,amount)
        elif code == "CSW": totals["withdrawals"] += abs(min(0.0,amount))
        elif code in DIVIDEND_TYPES: totals["gross_dividends"] += max(0.0,amount)
        elif code in WITHHOLDING_TYPES:
            totals["tax_withheld"] += abs(amount)
            if code.startswith("DIV"): totals["dividend_withheld"] += abs(amount)
        elif code == "INT": totals["interest"] += amount
        elif code in FEE_TYPES: totals["fees"] += abs(amount)
        elif code == "ACATC": totals["cash_transfers"] += amount
    realised, orphan_qty = _realised_fifo(fills)
    open_cost=sum(_number(p.get("cost_basis")) for p in positions)
    market_value=sum(_number(p.get("market_value")) for p in positions)
    unrealised=sum(_number(p.get("unrealized_pl")) for p in positions)
    equity=_number(account.get("equity") or account.get("portfolio_value"))
    cash=_number(account.get("cash"))
    net_flows=totals["deposits"]-totals["withdrawals"]+totals["cash_transfers"]
    summary={
        "current_equity":round(equity,2),"cash":round(cash,2),"buying_power":round(_number(account.get("buying_power")),2),
        "current_market_value":round(market_value,2),"current_invested_cost_basis":round(open_cost,2),
        "unrealized_pnl":round(unrealised,2),"realized_pnl_fifo":realised,"realized_fifo_unmatched_sell_qty":orphan_qty,
        "gross_lifetime_buys":round(totals["gross_buys"],2),"gross_lifetime_sells":round(totals["gross_sells"],2),
        "gross_dividends":round(totals["gross_dividends"],2),"dividend_withheld":round(totals["dividend_withheld"],2),
        "net_dividends":round(totals["gross_dividends"]-totals["dividend_withheld"],2),
        "tax_withheld":round(totals["tax_withheld"],2),"interest":round(totals["interest"],2),"fees":round(totals["fees"],2),
        "deposits":round(totals["deposits"],2),"withdrawals":round(totals["withdrawals"],2),
        "cash_transfers":round(totals["cash_transfers"],2),"net_cash_flows":round(net_flows,2),
        "equity_less_net_cash_flows":round(equity-net_flows,2),"activity_count":len(rows),"fill_count":len(fills)
    }
    rows.sort(key=lambda x:(x.get("date",""),x.get("id","")),reverse=True)
    return summary,rows

@account_history_bp.route("/api/analytics/account-history", methods=["GET"])
def account_history():
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    headers=_headers()
    if not headers["APCA-API-KEY-ID"] or not headers["APCA-API-SECRET-KEY"]:
        return jsonify({"error":"Alpaca API credentials are not configured."}),503
    date_from=request.args.get("after","").strip(); date_to=request.args.get("until","").strip()
    for value in (date_from,date_to):
        if value:
            try: datetime.strptime(value,"%Y-%m-%d")
            except ValueError: return jsonify({"error":"Dates must use YYYY-MM-DD."}),400
    if date_from and date_to and date_from>date_to:
        return jsonify({"error":"Start date must be on or before end date."}),400
    base=Config.ALPACA_BASE_URL.rstrip("/")
    try:
        ar=requests.get(base+"/v2/account",headers=headers,timeout=8); ar.raise_for_status(); account=ar.json()
        pr=requests.get(base+"/v2/positions",headers=headers,timeout=8); pr.raise_for_status(); positions=pr.json()
        activities=[]; page_token=None; has_more=False
        for page in range(MAX_PAGES):
            params={"activity_types":"FILL,TRANS,MISC,DIV,DIVCGL,DIVCGS,DIVNRA,DIVROC,DIVTXEX,DIVFT,DIVTW,DIVPWH","page_size":PAGE_SIZE,"direction":"desc"}
            if date_from: params["after"]=date_from+"T00:00:00Z"
            if date_to: params["until"]=date_to+"T23:59:59Z"
            if page_token: params["page_token"]=page_token
            response=requests.get(base+"/v2/account/activities",headers=headers,params=params,timeout=8)
            response.raise_for_status(); batch=response.json()
            if not isinstance(batch,list): raise ValueError("Alpaca account activities response was not a list.")
            if not batch: break
            activities.extend(batch)
            if len(batch)<PAGE_SIZE: break
            next_token=batch[-1].get("id")
            if not next_token or next_token==page_token: break
            page_token=next_token
            if page==MAX_PAGES-1: has_more=True
        summary,rows=build_history_summary(activities,positions,account)
        # A current equity snapshot cannot be compared to a partial-period cash-flow sum.
        if date_from or date_to or has_more:
            summary["equity_less_net_cash_flows"] = None
        return jsonify({
            "source":"Alpaca account activities","account_mode":Config.ALPACA_MODE,
            "summary":summary,"activities":rows,"has_more":has_more,
            "pages_loaded":min(MAX_PAGES,(len(activities)+PAGE_SIZE-1)//PAGE_SIZE),
            "date_from":date_from or None,"date_to":date_to or None,
            "notes":[
                "Current invested capital is the cost basis of positions still open.",
                "Buy/sell totals cover the selected date range and count gross executed trade value, including repeated turnover.",
                "Realised P&L is a long-only FIFO estimate from available fills; unmatched sells are excluded.",
                "Equity less net cash flows is an estimate and can be affected by securities transfers or incomplete account history.",
                "Broker activity records are not a tax statement; confirm tax treatment with official broker documents."
            ]
        })
    except requests.RequestException as exc:
        logger.warning("Alpaca account history request failed: %s",exc)
        return jsonify({"error":"Could not retrieve Alpaca account history.","detail":str(exc)[:240]}),502
    except Exception as exc:
        logger.exception("Account history calculation failed")
        return jsonify({"error":"Account history could not be calculated.","detail":str(exc)[:240]}),500
