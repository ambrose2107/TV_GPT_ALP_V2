"""
research/strategies/__init__.py
Importing this package registers every strategy module below into
STRATEGY_REGISTRY. To add a new strategy: create a new file in this folder
following the pattern in trend_pullback.py / ema_crossover.py /
rsi_meanreversion.py, then import it here.
"""
from .base import STRATEGY_REGISTRY, list_strategies, get_strategy  # noqa: F401

from . import trend_pullback   # noqa: F401
from . import ema_crossover    # noqa: F401
from . import rsi_meanreversion  # noqa: F401
