from datetime import datetime
import pytz

UTC = pytz.utc
JST = pytz.timezone("Asia/Tokyo")
EST = pytz.timezone("US/Eastern")


def utc_now():
    return datetime.now(UTC)


def format_all_timezones(dt=None):
    if dt is None:
        dt = utc_now()

    if dt.tzinfo is None:
        dt = UTC.localize(dt)

    return {
        "utc": dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "jst": dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
        "est": dt.astimezone(EST).strftime("%Y-%m-%d %H:%M:%S EST"),
    }
