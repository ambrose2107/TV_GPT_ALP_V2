#!/usr/bin/env python3
"""Download Dukascopy XAUUSD M1 candles and import them into the local data library.

The downloader fetches one UTC day at a time from Dukascopy's public datafeed
endpoint. It never commits raw or normalized market data to Git. Run locally:
  python scripts/download_dukascopy_xauusd.py --from 2003-05-05 --price-side BID

Dukascopy's official bulk export may instead require S3 Requester Pays credentials.
This script uses the public daily BI5 candle endpoint; availability can vary.
"""
from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import json
import lzma
import logging
import struct
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.xauusd_data_pipeline import import_files, validate_library

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SYMBOL = "XAUUSD"
PRICE_SCALE = 1000.0
RECORD = struct.Struct(">5If")  # seconds, open, close, low, high, volume
LOG = logging.getLogger("dukascopy_xauusd")


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=0,
        status=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "OptiTrade-XAUUSD-Research/1.0"})
    return session


def decode_bi5(payload: bytes, day: date) -> pd.DataFrame:
    """Decode one Dukascopy daily M1 BI5 file to UTC OHLCV rows."""
    if not payload:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    raw = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    if len(raw) % RECORD.size:
        raise ValueError(
            f"Malformed BI5 payload for {day}: {len(raw)} bytes is not divisible "
            f"by {RECORD.size}"
        )
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rows = []
    for offset in range(0, len(raw), RECORD.size):
        seconds, open_, close, low, high, volume = RECORD.unpack_from(raw, offset)
        if seconds >= 86400:
            raise ValueError(f"Invalid seconds-from-midnight value {seconds} for {day}")
        rows.append((
            midnight + timedelta(seconds=seconds),
            open_ / PRICE_SCALE,
            high / PRICE_SCALE,
            low / PRICE_SCALE,
            close / PRICE_SCALE,
            float(volume),
        ))
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    if frame["time"].duplicated().any():
        raise ValueError(f"Duplicate minute timestamps in BI5 file for {day}")
    return frame.sort_values("time")


def download_day(day: date, side: str, session: requests.Session) -> pd.DataFrame | None:
    # Dukascopy URL month component is zero-based.
    url = (
        f"{BASE_URL}/{SYMBOL}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{side}_candles_min_1.bi5"
    )
    # The public Dukascopy feed has recently exhibited intermittent stalls.
    # Retry each day explicitly with a long read timeout; never treat a timeout
    # as an empty/weekend file.
    last_error = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(15, 90))
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return decode_bi5(response.content, day)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == 3:
                break
            delay = min(3 * (2 ** attempt), 24)
            LOG.warning("Dukascopy request for %s failed (%s); retry %s/3 in %ss",
                        day, exc.__class__.__name__, attempt + 1, delay)
            time.sleep(delay)
    raise RuntimeError(
        f"Dukascopy timed out or failed for {day} after 4 attempts. "
        "The provider feed is intermittently unavailable; retry this date later."
    ) from last_error


def _fetch_one(args):
    day, side = args
    session = build_session()
    try:
        frame = download_day(day, side, session)
        return day, frame, None
    except Exception as exc:  # returned to main thread for a clear failure summary
        return day, None, str(exc)
    finally:
        session.close()


def download_range(start: date, end: date, side: str, workers: int = 1) -> tuple[list[pd.DataFrame], int]:
    if end < start:
        raise ValueError("--to must be on or after --from")
    if side not in {"BID", "ASK"}:
        raise ValueError("price side must be BID or ASK")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    frames: list[pd.DataFrame] = []
    missing = 0
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:
        futures = [pool.submit(_fetch_one, (day, side)) for day in days]
        for future in concurrent.futures.as_completed(futures):
            day, frame, error = future.result()
            if error:
                failures.append((day.isoformat(), error))
            elif frame is None or frame.empty:
                missing += 1
            else:
                frames.append(frame)
    if failures:
        sample = "; ".join(f"{d}: {e}" for d, e in failures[:8])
        raise RuntimeError(f"{len(failures)} daily download(s) failed; examples: {sample}")
    frames.sort(key=lambda f: f["time"].iloc[0])
    return frames, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True,
                        help="First UTC date, YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date(),
                        help="Last UTC date inclusive (default: today)")
    parser.add_argument("--price-side", choices=("BID", "ASK"), default="BID")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("market_data/XAUUSD"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Downloading %s XAUUSD M1 candles: %s to %s UTC",
             args.price_side, args.start, args.end)
    total_bars = 0
    missing_days = 0
    first_utc = None
    last_utc = None
    outputs = {tf: [] for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1")}
    cursor = args.start.replace(day=1)
    args.out.mkdir(parents=True, exist_ok=True)

    # Process one calendar month at a time so a 20+ year download stays memory-bounded.
    with tempfile.TemporaryDirectory(prefix="optitrade_xauusd_") as tmp:
        while cursor <= args.end:
            month_last = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1])
            month_start = max(args.start, cursor)
            month_end = min(args.end, month_last)
            LOG.info("Fetching %s through %s", month_start, month_end)
            frames, missing = download_range(month_start, month_end, args.price_side, args.workers)
            missing_days += missing
            if frames:
                frame = pd.concat(frames, ignore_index=True).sort_values("time")
                frame = frame.drop_duplicates("time", keep="last")
                source_file = Path(tmp) / f"dukascopy_{cursor.year:04d}_{cursor.month:02d}.csv"
                frame.to_csv(source_file, index=False)
                month_meta = import_files(
                    [source_file], "dukascopy", args.out, "UTC", "M1", args.price_side
                )
                total_bars += month_meta["bars"]
                first_utc = first_utc or month_meta["start_utc"]
                last_utc = month_meta["end_utc"]
                for tf, entries in month_meta["outputs"].items():
                    outputs.setdefault(tf, []).extend(entries)
            cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)

    if total_bars == 0:
        raise RuntimeError("No candle files were returned; no dataset was written")
    root = args.out / args.price_side
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "instrument": SYMBOL,
        "source": "dukascopy",
        "source_timeframe": "M1",
        "canonical_timezone": "UTC",
        "price_side": args.price_side,
        "requested_start_utc": args.start.isoformat(),
        "requested_end_utc": args.end.isoformat(),
        "start_utc": first_utc,
        "end_utc": last_utc,
        "bars": total_bars,
        "missing_or_empty_daily_files": missing_days,
        "volume_semantics": "provider tick volume; not centralized exchange volume",
        "outputs": outputs,
    }
    (metadata_dir / "latest_import.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = validate_library(root)
    if report["status"] != "PASS":
        raise RuntimeError(f"Post-import library validation failed: {report}")
    LOG.info(
        "PASS: %s bars, %s to %s UTC; %s empty/missing daily files; outputs: %s",
        total_bars, first_utc, last_utc, missing_days, args.out
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
