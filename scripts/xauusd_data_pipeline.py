#!/usr/bin/env python3
"""Normalize provider-supplied XAUUSD OHLC files for reproducible research."""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, re, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/"market_data"/"XAUUSD"
RULES={"M1":"1min","M5":"5min","M15":"15min","M30":"30min","H1":"1h","H4":"4h","D1":"1D"}
ALIASES={"timestamp":"time","datetime":"time","date_time":"time","time_utc":"time",
"open":"open","o":"open","high":"high","h":"high","low":"low","l":"low",
"close":"close","c":"close","volume":"volume","vol":"volume","tickvol":"volume",
"tick_volume":"volume","gmt_time":"time"}

def _read_csv_text(text):
    first=next((line for line in text.splitlines() if line.strip()), "")
    delimiter=";" if ";" in first else ","
    first_field=first.split(delimiter,1)[0].strip()
    # HistData Generic ASCII M1 is headerless: YYYYMMDD HHMMSS;O;H;L;C;V.
    if re.fullmatch(r"\d{8}\s+\d{6}", first_field) and len(first.split(delimiter)) >= 5:
        names=["time","open","high","low","close","volume"]
        return pd.read_csv(io.StringIO(text),sep=delimiter,header=None,names=names,engine="python")
    return pd.read_csv(io.StringIO(text),sep=None,engine="python")

def read_file(path):
    if path.suffix.lower()==".zip":
        frames=[]
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith((".csv",".txt")):
                    with z.open(name) as f:
                        frames.append(_read_csv_text(f.read().decode("utf-8-sig")))
        if not frames: raise ValueError("ZIP contains no CSV/TXT files")
        return pd.concat(frames,ignore_index=True)
    if path.suffix.lower()==".gz":
        with gzip.open(path,"rt",encoding="utf-8-sig") as f: return _read_csv_text(f.read())
    return _read_csv_text(path.read_text(encoding="utf-8-sig"))

def normalize_frame(raw, source, source_timezone=None):
    df=raw.copy()
    rename={}
    for col in df.columns:
        key=re.sub(r"[\s\-]+","_",str(col).strip().lower())
        rename[col]=ALIASES.get(key,key)
    df=df.rename(columns=rename)
    if "time" not in df.columns and "date" in df.columns:
        candidates=[c for c in df.columns if c not in {"date","open","high","low","close","volume"}]
        if candidates: df=df.rename(columns={candidates[0]:"time"})
    if not {"open","high","low","close"}.issubset(df.columns):
        raise ValueError("OHLC columns missing; expected Open, High, Low, Close or O/H/L/C")
    if "time" not in df.columns: raise ValueError("Missing timestamp; expected time/timestamp or Date + Time")
    raw_time=(df["date"].astype(str)+" "+df["time"].astype(str)) if "date" in df.columns else df["time"]
    # HistData's headerless ASCII M1 uses compact YYYYMMDD HHMMSS values.
    # Parse that format explicitly so pandas parser-version inference cannot
    # silently turn valid timestamps into NaT.
    if str(source).lower() == "histdata":
        compact_time = raw_time.astype(str).str.strip()
        if compact_time.str.fullmatch(r"\d{8}\s+\d{6}").all():
            parsed=pd.to_datetime(compact_time,format="%Y%m%d %H%M%S",errors="coerce")
        else:
            parsed=pd.to_datetime(raw_time,errors="coerce")
    else:
        parsed=pd.to_datetime(raw_time,errors="coerce")
    if getattr(parsed.dt,"tz",None) is None:
        if not source_timezone: raise ValueError("Naive timestamps require --source-timezone (for example UTC)")
        parsed=parsed.dt.tz_localize(source_timezone,ambiguous="NaT",nonexistent="NaT")
    df["time"]=parsed.dt.tz_convert("UTC")
    for col in ("open","high","low","close"): df[col]=pd.to_numeric(df[col],errors="coerce")
    if "volume" not in df: df["volume"]=0.0
    df["volume"]=pd.to_numeric(df["volume"],errors="coerce").fillna(0.0)
    df=df.dropna(subset=["time","open","high","low","close"])
    df=df[(df[["open","high","low","close"]]>0).all(axis=1)]
    bad=(df["high"]<df[["open","close","low"]].max(axis=1)) | (df["low"]>df[["open","close","high"]].min(axis=1))
    if bad.any(): raise ValueError(f"{int(bad.sum())} rows violate OHLC high/low constraints")
    df=df[["time","open","high","low","close","volume"]].sort_values("time").drop_duplicates("time",keep="last").set_index("time")
    df.index.name="time"
    if df.empty: raise ValueError("No valid OHLC rows after normalization")
    return df

def resample_ohlc(df, rule):
    return df.resample(rule,label="left",closed="left",origin="start_day").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","high","low","close"])

def write_monthly(df, timeframe, out_dir):
    outputs=[]
    for (year,month),part in df.groupby([df.index.year,df.index.month]):
        folder=out_dir/"normalized"/timeframe/f"{year:04d}"
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"XAUUSD_{timeframe}_{year:04d}_{month:02d}.csv.gz"
        part=part.copy()
        part.index.name="time"
        part.to_csv(path,compression="gzip",float_format="%.5f")
        outputs.append({"path":str(path.relative_to(out_dir)),"bars":len(part),
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "start_utc":part.index[0].isoformat(),"end_utc":part.index[-1].isoformat()})
    return outputs

def import_files(files,source,out_dir,source_timezone,source_timeframe,price_side="UNKNOWN"):
    if price_side not in {"BID","ASK","MID","UNKNOWN"}: raise ValueError("price_side must be BID, ASK, MID or UNKNOWN")
    dataset_dir=out_dir/price_side if price_side!="UNKNOWN" else out_dir
    frames=[normalize_frame(read_file(p),source,source_timezone) for p in files]
    base=pd.concat(frames).sort_index()
    base=base[~base.index.duplicated(keep="last")]
    if source_timeframe not in RULES: raise ValueError("Unsupported source timeframe")
    written={source_timeframe:write_monthly(base,source_timeframe,dataset_dir)}
    if source_timeframe=="M1":
        for tf in ("M5","M15","M30","H1","H4","D1"):
            written[tf]=write_monthly(resample_ohlc(base,RULES[tf]),tf,dataset_dir)
    meta={"instrument":"XAUUSD","source":source,"source_timeframe":source_timeframe,
          "canonical_timezone":"UTC","source_timezone":source_timezone,
          "price_side":price_side,
          "volume_semantics":"provider-specific; commonly tick volume, not centralized exchange volume",
          "imported_at_utc":datetime.now(timezone.utc).isoformat(),"source_files":[str(p) for p in files],
          "bars":len(base),"start_utc":base.index[0].isoformat(),"end_utc":base.index[-1].isoformat(),"outputs":written}
    (dataset_dir/"metadata").mkdir(parents=True,exist_ok=True)
    (dataset_dir/"metadata"/"latest_import.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta

def validate_library(out_dir):
    reports=[]
    for folder in sorted((out_dir/"normalized").glob("*")):
        if not folder.is_dir(): continue
        files=sorted(folder.glob("*/*.csv.gz")); total=0; last=None
        for path in files:
            df=pd.read_csv(path,compression="gzip",parse_dates=["time"])
            if df.empty: raise ValueError(f"Empty file: {path}")
            idx=pd.to_datetime(df["time"],utc=True)
            if idx.duplicated().any() or not idx.is_monotonic_increasing: raise ValueError(f"Duplicate/unsorted timestamps: {path}")
            if last is not None and idx.iloc[0]<=last: raise ValueError(f"Overlapping monthly partitions: {folder.name}")
            for c in ("open","high","low","close"):
                if df[c].isna().any() or (df[c]<=0).any(): raise ValueError(f"Invalid {c}: {path}")
            if (df["high"]<df[["open","close","low"]].max(axis=1)).any(): raise ValueError(f"Invalid high: {path}")
            if (df["low"]>df[["open","close","high"]].min(axis=1)).any(): raise ValueError(f"Invalid low: {path}")
            last=idx.iloc[-1]; total+=len(df)
        reports.append({"timeframe":folder.name,"files":len(files),"bars":total})
    report={"instrument":"XAUUSD","checked_at_utc":datetime.now(timezone.utc).isoformat(),
            "timeframes":reports,"status":"PASS" if reports else "EMPTY"}
    (out_dir/"metadata").mkdir(parents=True,exist_ok=True)
    (out_dir/"metadata"/"quality_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser(description="Import and validate XAUUSD historical OHLC data")
    sub=p.add_subparsers(dest="cmd",required=True)
    i=sub.add_parser("import"); i.add_argument("files",nargs="+",type=Path)
    i.add_argument("--source",choices=("dukascopy","forexsb","histdata","generic"),required=True)
    i.add_argument("--source-timeframe",choices=tuple(RULES),default="M1")
    i.add_argument("--source-timezone",default=None); i.add_argument("--price-side",choices=("BID","ASK","MID","UNKNOWN"),default="UNKNOWN"); i.add_argument("--out",type=Path,default=DEFAULT_OUT)
    v=sub.add_parser("validate"); v.add_argument("--price-side",choices=("BID","ASK","MID","UNKNOWN"),default="UNKNOWN"); v.add_argument("--out",type=Path,default=DEFAULT_OUT)
    a=p.parse_args()
    try:
        root=a.out/a.price_side if a.price_side!="UNKNOWN" else a.out
        result=import_files(a.files,a.source,a.out,a.source_timezone,a.source_timeframe,a.price_side) if a.cmd=="import" else validate_library(root)
        print(json.dumps(result,indent=2)); return 0
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
