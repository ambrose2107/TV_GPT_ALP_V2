"""
core/scheduler.py
Runs research/signal_engine.run_full_scan() on a background interval
(3h / 6h / 12h / off), plus supports an immediate "run now" trigger.

Uses APScheduler's BackgroundScheduler, which runs jobs in its own thread
pool -- separate from the gunicorn worker thread handling HTTP requests/
webhooks, so a multi-minute scan never freezes trade execution.

Only one scan runs at a time (a lock guards against overlapping runs if
"Run Now" is clicked while a scheduled scan is already in progress).
"""
import threading
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from core.logger import get_logger
from core.database import get_setting, set_setting
from research.signal_engine import run_full_scan

logger = get_logger(__name__)

_scheduler = BackgroundScheduler(daemon=True)
_scan_lock = threading.Lock()
_JOB_ID = "signal_scan_job"

VALID_INTERVALS = {0: "Off", 3: "Every 3h", 6: "Every 6h", 12: "Every 12h"}


def _run_scan_guarded(symbols: list = None):
    if not _scan_lock.acquire(blocking=False):
        logger.info("scheduler: scan already in progress, skipping this trigger")
        return
    try:
        set_setting("last_scan_started_at", datetime.utcnow().isoformat() + "Z")
        summary = run_full_scan(symbols=symbols)
        set_setting("last_scan_summary", str(summary))
        set_setting("last_scan_finished_at", datetime.utcnow().isoformat() + "Z")
    except Exception as e:
        logger.error(f"scheduler: scan failed: {e}")
        set_setting("last_scan_summary", f"error: {e}")
    finally:
        _scan_lock.release()


def run_now_async(symbols: list = None):
    """Kick off a scan immediately in a background thread, return instantly."""
    if _scan_lock.locked():
        return {"started": False, "reason": "A scan is already running"}
    t = threading.Thread(target=_run_scan_guarded, kwargs={"symbols": symbols}, daemon=True)
    t.start()
    return {"started": True}


def set_interval_hours(hours: int):
    hours = int(hours)
    if hours not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {list(VALID_INTERVALS)}")
    set_setting("scan_interval_hours", hours)

    if _scheduler.get_job(_JOB_ID):
        _scheduler.remove_job(_JOB_ID)
    if hours > 0:
        _scheduler.add_job(_run_scan_guarded, "interval", hours=hours, id=_JOB_ID,
                            next_run_time=datetime.now() + timedelta(seconds=10))
    logger.info(f"scheduler: interval set to {VALID_INTERVALS[hours]}")


def get_status() -> dict:
    hours = int(get_setting("scan_interval_hours", 0) or 0)
    job = _scheduler.get_job(_JOB_ID)
    return {
        "interval_hours": hours,
        "interval_label": VALID_INTERVALS.get(hours, "Off"),
        "next_run_at": job.next_run_time.isoformat() if job and job.next_run_time else None,
        "last_scan_started_at": get_setting("last_scan_started_at"),
        "last_scan_finished_at": get_setting("last_scan_finished_at"),
        "last_scan_summary": get_setting("last_scan_summary"),
        "scan_in_progress": _scan_lock.locked(),
    }


def init_scheduler():
    """Call once at app startup. Restores whatever interval was last saved."""
    if not _scheduler.running:
        _scheduler.start()
    saved_hours = int(get_setting("scan_interval_hours", 0) or 0)
    if saved_hours > 0:
        set_interval_hours(saved_hours)
    logger.info(f"scheduler: initialized, interval={VALID_INTERVALS.get(saved_hours,'Off')}")
