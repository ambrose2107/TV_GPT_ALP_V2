"""Unified Gold Strategy Lab.
Runs the four existing XAUUSD research strategies from one authenticated page:
V3 institutional confluence, Fib MTF, Confluence V4, and Pullback V1.
"""
from flask import Blueprint, render_template, session

xauusd_strategy_lab_bp = Blueprint("xauusd_strategy_lab", __name__)


@xauusd_strategy_lab_bp.route("/xauusd-strategy-lab")
def xauusd_strategy_lab_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_strategy_lab.html")
