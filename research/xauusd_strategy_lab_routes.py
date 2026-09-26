"""Strategy Lab routes.

The stable/original seven-strategy lab is exposed as V2 because it is the
known-working workspace. The newer three-strategy research lab is exposed as
V1 so the two can be run independently without changing their backend engines.
"""
from flask import Blueprint, render_template, session, redirect, url_for

xauusd_strategy_lab_bp = Blueprint("xauusd_strategy_lab", __name__)

def _auth_page(template):
    if not session.get("logged_in"):
        return redirect(url_for("dashboard.login"))
    return render_template(template)

@xauusd_strategy_lab_bp.route("/xauusd-strategy-lab")
@xauusd_strategy_lab_bp.route("/xauusd-strategy-lab-v2")
def xauusd_strategy_lab_v2_page():
    return _auth_page("xauusd_strategy_lab.html")

@xauusd_strategy_lab_bp.route("/xauusd-strategy-lab-v1")
def xauusd_strategy_lab_v1_page():
    return _auth_page("xauusd_strategy_lab_v2.html")
