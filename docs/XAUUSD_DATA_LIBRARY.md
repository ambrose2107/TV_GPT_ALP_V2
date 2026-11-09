# XAUUSD Historical Data Library

This is the canonical historical research-data area for OptiTrade. It is separate from live Alpaca feeds.

## Approved acquisition workflow

Use provider-owned download methods. This repository does not scrape websites, bypass logins, or assume redistribution rights.

- Dukascopy's official Historical Data Export provides historical instrument data with selectable instrument, candle/tick type, price side, date range and timezone. XAUUSD is available through its historical-data export.
- ForexSB's Historical Forex Data service explicitly lists XAUUSD and says its bar data is compiled from Dukascopy tick data. It offers precompiled M1, M5, M15 and M30 periods.
- HistData should be used only after confirming XAUUSD is actually available for the required period and timeframe. Its Generic ASCII M1 format is headerless and semicolon-delimited (`YYYYMMDD HHMMSS;Open;High;Low;Close;Volume`). Its timestamps are fixed EST (UTC-5) without daylight-saving adjustment; import with `--source-timezone Etc/GMT+5`, not `America/New_York`. Do not assume every symbol is supported.
- dukascopy-node is a third-party downloader, not an official Dukascopy application. Treat it as an optional acquisition helper and inspect its options and output before use.

Provider references:
- https://www.dukascopy.com/swiss/english/marketwatch/historical/
- https://forexsb.com/historical-forex-data
- https://forexsb.com/wiki/fsbpro_guide/data_download
- https://www.dukascopy-node.app/instrument/xauusd
- https://www.histdata.com/download-free-forex-historical-data/

## Import and validate

The importer uses pandas, already present in requirements.txt. It accepts CSV, TXT, CSV.GZ and ZIP files containing CSV/TXT. Files must have OHLC columns and a combined timestamp or Date + Time.

Example commands:

    python scripts/xauusd_data_pipeline.py import ./downloads/xauusd_bid_m1.csv --source dukascopy --source-timeframe M1 --source-timezone UTC --price-side BID\n\n    # HistData Generic ASCII M1 (fixed EST, no DST):\n    python scripts/xauusd_data_pipeline.py import ./downloads/DAT_ASCII_XAUUSD_M1_202401.csv --source histdata --source-timeframe M1 --source-timezone Etc/GMT+5 --price-side BID

    python scripts/xauusd_data_pipeline.py validate --price-side BID

For ForexSB or HistData, change the source name. If timestamps are timezone-naive, supply the timezone selected during export. Never guess the timezone. If a provider offers BID and ASK files, import and retain them separately; never mix price sides silently.

## Dataset layout

Normalized files are stored as monthly gzip-compressed CSV partitions. BID, ASK and MID are kept in separate subfolders so importing a second price side cannot overwrite the first:

    market_data/XAUUSD/BID/
      normalized/M1/YYYY/XAUUSD_M1_YYYY_MM.csv.gz
      normalized/M5/YYYY/XAUUSD_M5_YYYY_MM.csv.gz
      metadata/latest_import.json
      metadata/quality_report.json
    market_data/XAUUSD/ASK/ (separate dataset when available)

M1 data can be aggregated into M5/M15/M30/H1/H4/D1 using UTC clock boundaries. Coarser source bars are never expanded into fabricated M1 data. Missing bars are not forward-filled. Monthly partitions include SHA-256 checksums in the import manifest.

## GitHub storage and licensing

Commit monthly compressed partitions only after checking the provider's terms for redistribution and the repository's visibility. Do not commit huge raw tick archives, credentials or personal account downloads. If redistribution is prohibited, keep source files locally and commit only permitted derived outputs or metadata. For large permitted datasets, consider Git LFS or GitHub Release assets rather than ordinary Git blobs. Keep a source, price-side and timezone note with each dataset.

## Research caveats

XAUUSD is an OTC market, not one consolidated exchange tape. Provider and broker prices can differ because of bid/ask side, spread, rollover, daylight-saving rules and trading sessions. Forex volume is usually tick volume, not centralized exchange volume. Backtests should record source, price side, timezone, timeframe, spread/slippage assumptions and dataset checksum. Compare overlapping periods from at least two sources before treating a strategy result as robust.
