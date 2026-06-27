"""
core/excel_export.py — Export trades to Excel with formatting
"""
import io
import pandas as pd
from datetime import datetime
from core.database import get_all_trades, get_all_closed_positions
from core.logger import get_logger

logger = get_logger(__name__)

def export_trades_excel() -> bytes:
    """Export all trades + closed positions to formatted Excel file."""
    trades  = get_all_trades()
    closed  = get_all_closed_positions()

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # ── Sheet 1: All Trades ───────────────────────────────────────────
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.to_excel(writer, sheet_name="All Trades", index=False)
            ws = writer.sheets["All Trades"]
            _format_sheet(ws, df_trades)
        else:
            pd.DataFrame({"Note": ["No trades yet"]}).to_excel(
                writer, sheet_name="All Trades", index=False)

        # ── Sheet 2: Closed Positions ─────────────────────────────────────
        if closed:
            df_closed = pd.DataFrame(closed)
            df_closed.to_excel(writer, sheet_name="Closed Positions", index=False)
            ws2 = writer.sheets["Closed Positions"]
            _format_sheet(ws2, df_closed)
        else:
            pd.DataFrame({"Note": ["No closed positions yet"]}).to_excel(
                writer, sheet_name="Closed Positions", index=False)

        # ── Sheet 3: P&L Summary ──────────────────────────────────────────
        if closed:
            df_c    = pd.DataFrame(closed)
            winners = df_c[df_c["pnl"] > 0]["pnl"].sum() if "pnl" in df_c else 0
            losers  = df_c[df_c["pnl"] < 0]["pnl"].sum() if "pnl" in df_c else 0
            total   = df_c["pnl"].sum() if "pnl" in df_c else 0
            win_rate= (len(df_c[df_c["pnl"] > 0]) / len(df_c) * 100) if len(df_c) > 0 else 0

            summary = pd.DataFrame([
                {"Metric": "Total Trades",    "Value": len(closed)},
                {"Metric": "Winning Trades",  "Value": len(df_c[df_c["pnl"] > 0]) if "pnl" in df_c else 0},
                {"Metric": "Losing Trades",   "Value": len(df_c[df_c["pnl"] < 0]) if "pnl" in df_c else 0},
                {"Metric": "Win Rate %",      "Value": round(win_rate, 1)},
                {"Metric": "Total P&L ($)",   "Value": round(total, 2)},
                {"Metric": "Gross Profit ($)","Value": round(winners, 2)},
                {"Metric": "Gross Loss ($)",  "Value": round(losers, 2)},
                {"Metric": "Best Trade ($)",  "Value": round(float(df_c["pnl"].max()), 2) if "pnl" in df_c else 0},
                {"Metric": "Worst Trade ($)", "Value": round(float(df_c["pnl"].min()), 2) if "pnl" in df_c else 0},
                {"Metric": "Avg P&L ($)",     "Value": round(float(df_c["pnl"].mean()), 2) if "pnl" in df_c else 0},
                {"Metric": "Export Date",     "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
            ])
            summary.to_excel(writer, sheet_name="P&L Summary", index=False)
        else:
            pd.DataFrame({"Note": ["No closed positions yet"]}).to_excel(
                writer, sheet_name="P&L Summary", index=False)

    output.seek(0)
    return output.read()


def _format_sheet(ws, df):
    """Auto-size columns."""
    for i, col in enumerate(df.columns, 1):
        try:
            max_len = max(len(str(col)), df[col].astype(str).str.len().max() if len(df) > 0 else 0)
        except:
            max_len = len(str(col)) + 4
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max_len + 4, 40)
