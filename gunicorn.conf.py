# V4 optimizer/backtests can legitimately run longer than Gunicorn's 30s default.
timeout = 180
graceful_timeout = 25
keepalive = 5
workers = 1
