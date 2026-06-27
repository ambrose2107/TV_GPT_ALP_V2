"""
core/timezone_utils.py
Proper timezone handling using pytz.
UTC, JST (Asia/Tokyo), EST/EDT (US/Eastern)
"""
try:
    import pytz
    _HAS_PYTZ = True
    UTC = pytz.utc
    JST = pytz.timezone("Asia/Tokyo")
    EST = pytz.timezone("US/Eastern")
except ImportError:
    _HAS_PYTZ = False
    from datetime import timezone, timedelta
    UTC = timezone.utc
    JST = timezone(timedelta(hours=9),  "JST")
    EST = timezone(timedelta(hours=-5), "EST")

from datetime import datetime


def now_all_zones() -> dict:
    """Current time in UTC, JST, and US/Eastern with proper DST."""
    if _HAS_PYTZ:
        now_utc = datetime.now(UTC)
        now_jst = now_utc.astimezone(JST)
        now_est = now_utc.astimezone(EST)
        tz_label = now_est.strftime("%Z")  # EDT or EST
        offset   = now_est.strftime("%z")
        return {
            "utc":      now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "jst":      now_jst.strftime("%Y-%m-%d %H:%M:%S JST"),
            "ny":       now_est.strftime(f"%Y-%m-%d %H:%M:%S {tz_label} (UTC{offset[:3]}:{offset[3:]})"),
            "tz_label": tz_label,
            "is_dst":   tz_label == "EDT",
        }
    else:
        # Fallback without pytz
        from datetime import timezone, timedelta
        now = datetime.now(timezone.utc)
        jst = now.astimezone(timezone(timedelta(hours=9)))
        # Approx DST
        y = now.year
        mar = datetime(y, 3, 1, tzinfo=timezone.utc)
        dst_start = mar + __import__('datetime').timedelta(days=(6-mar.weekday())%7+7)
        nov = datetime(y, 11, 1, tzinfo=timezone.utc)
        dst_end = nov + __import__('datetime').timedelta(days=(6-nov.weekday())%7)
        is_dst = dst_start <= now < dst_end
        off = timedelta(hours=-4 if is_dst else -5)
        ny  = now.astimezone(timezone(off))
        tz_label = "EDT" if is_dst else "EST"
        return {
            "utc":      now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "jst":      jst.strftime("%Y-%m-%d %H:%M:%S JST"),
            "ny":       ny.strftime(f"%Y-%m-%d %H:%M:%S {tz_label}"),
            "tz_label": tz_label,
            "is_dst":   is_dst,
        }


def market_sessions_utc() -> dict:
    """
    All market session times in UTC, JST, and NY.
    Accounts for current DST status.
    """
    t   = now_all_zones()
    dst = t["is_dst"]
    ny_off   = -4 if dst else -5
    tz_label = "EDT" if dst else "EST"

    # Session definitions in UTC
    raw_sessions = {
        "Asia_Open":       {"utc": "00:00", "desc": "Tokyo open"},
        "Asia_Close":      {"utc": "08:00", "desc": "Tokyo close"},
        "London_Open":     {"utc": "08:00", "desc": "London open"},
        "London_Close":    {"utc": "17:00", "desc": "London close"},
        "NY_PreMarket":    {"utc": "08:00", "desc": "US pre-market"},
        "NY_Open":         {"utc": "13:30" if dst else "14:30", "desc": "NYSE open"},
        "NY_Close":        {"utc": "20:00" if dst else "21:00", "desc": "NYSE close"},
        "NY_AfterHours":   {"utc": "20:00" if dst else "21:00", "desc": "After-hours end"},
    }

    sessions = {}
    for name, info in raw_sessions.items():
        h, m  = map(int, info["utc"].split(":"))
        jst_h = (h + 9) % 24
        ny_h  = (h + ny_off) % 24
        sessions[name] = {
            "utc":  f"{h:02d}:{m:02d}",
            "jst":  f"{jst_h:02d}:{m:02d}",
            "ny":   f"{ny_h:02d}:{m:02d}",
            "desc": info["desc"],
        }

    # Grouped session ranges
    session_ranges = {
        "Asia Session":     {"open_utc":"00:00","close_utc":"08:00"},
        "London Session":   {"open_utc":"08:00","close_utc":"17:00"},
        "NY Pre-Market":    {"open_utc":"08:00","close_utc": "13:30" if dst else "14:30"},
        "NY Regular Hours": {"open_utc": "13:30" if dst else "14:30",
                             "close_utc":"20:00" if dst else "21:00"},
        "NY After-Hours":   {"open_utc":"20:00" if dst else "21:00","close_utc":"24:00"},
        "Overlap London+NY":{"open_utc": "13:30" if dst else "14:30","close_utc":"17:00"},
    }

    ranges_formatted = {}
    for name, r in session_ranges.items():
        def fmt_range(utc_str):
            h,m = map(int, utc_str.split(":"))
            if h >= 24: return "00:00"
            return f"{h:02d}:{m:02d}"
        def to_jst(utc_str):
            h,m = map(int, utc_str.split(":"))
            return f"{(h+9)%24:02d}:{m:02d}"
        def to_ny(utc_str):
            h,m = map(int, utc_str.split(":"))
            if h >= 24: return "00:00"
            return f"{(h+ny_off)%24:02d}:{m:02d}"
        ranges_formatted[name] = {
            "utc":  f"{fmt_range(r['open_utc'])} – {fmt_range(r['close_utc'])}",
            "jst":  f"{to_jst(r['open_utc'])} – {to_jst(r['close_utc'])}",
            "ny":   f"{to_ny(r['open_utc'])} – {to_ny(r['close_utc'])}",
        }

    return {
        "now":           t,
        "tz_label":      tz_label,
        "ny_offset":     ny_off,
        "events":        sessions,
        "sessions":      ranges_formatted,
    }


def alpaca_ts_to_all_zones(ts_str: str) -> dict:
    """
    Convert an Alpaca timestamp string (UTC) to all 3 timezone displays.
    ts_str: ISO format like '2026-05-22T14:30:00Z'
    """
    try:
        ts_str = ts_str.replace("Z", "+00:00")
        if _HAS_PYTZ:
            dt_utc = datetime.fromisoformat(ts_str).astimezone(UTC)
            return {
                "utc": dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
                "jst": dt_utc.astimezone(JST).strftime("%Y-%m-%d %H:%M JST"),
                "ny":  dt_utc.astimezone(EST).strftime("%Y-%m-%d %H:%M %Z"),
            }
        else:
            from datetime import timezone, timedelta
            dt = datetime.fromisoformat(ts_str)
            t  = now_all_zones()
            off = timedelta(hours=-4 if t["is_dst"] else -5)
            return {
                "utc": dt.strftime("%Y-%m-%d %H:%M UTC"),
                "jst": dt.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST"),
                "ny":  dt.astimezone(timezone(off)).strftime(f"%Y-%m-%d %H:%M {'EDT' if t['is_dst'] else 'EST'}"),
            }
    except Exception as e:
        return {"utc": ts_str, "jst": "", "ny": ""}
