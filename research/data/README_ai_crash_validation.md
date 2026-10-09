# AI Crash Historical Validation Dataset

Purpose: fixed historical baseline for retrospective crash validation only. Live dashboard scoring must continue to use the live market-data pipeline.

## Data
- `ai_crash_validation_history.csv`: daily date, SPY close, QQQ adjusted close (where provided), and VIX close.
- SPY source: OStochastic/Daily-SPY-data-from-2000-2025 public GitHub dataset (Yahoo Finance-derived; header rows normalized).
- QQQ source: airinthespace/stock-price-analysis QQQ_raw.csv (Yahoo Finance-derived; adjusted_close used where present).
- VIX source: datasets/finance-vix, sourced from Cboe VIX daily history.

## Important limitations
- These are third-party public historical copies, not a licensed institutional feed. Verify upstream terms before redistributing publicly.
- Source histories have different end dates. Blank values are missing observations, never interpolated or fabricated.
- This baseline is for validation, not production quotes. New recent rows must be fetched and merged by date from the live data provider before declaring current coverage.
- Before validation is trusted, the replay must report per-symbol coverage and require valid inputs for each episode. VIX's methodology changed in 2003; treat pre-2003 values as the historical series available at the time.
- Keep this dataset small and avoid committing intraday or portfolio-specific data.
