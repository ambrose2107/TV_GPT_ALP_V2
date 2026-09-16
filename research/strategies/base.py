"""
research/strategies/base.py
Strategy registry for the Backtest tab.

To add a new strategy to the dashboard: create a new file in this folder
that defines a STRATEGY dict (see the examples in this folder) and register
it in registry.py. No server restart of any "code executor" is needed and
no arbitrary code is ever eval'd/exec'd from the web UI — strategies are
plain Python, reviewed and committed like any other code in this repo.

A strategy is a plain dict:
{
    "id": "unique_snake_case_id",
    "name": "Human readable name",
    "description": "One-line description shown in the UI",
    "default_params": { "param_name": default_value, ... },
    "param_schema": { "param_name": {"type": "int"|"float"|"bool", "min":..., "max":..., "step":...}, ... },
    "generate_signals": function(df, params) -> df with a "signal" column
                        (1 = long entry, -1 = short entry, 0 = none),
                        computed WITHOUT lookahead (signal at bar i must
                        only use data up to and including bar i).
}
"""

STRATEGY_REGISTRY = {}


def register(strategy: dict):
    sid = strategy["id"]
    STRATEGY_REGISTRY[sid] = strategy
    return strategy


def list_strategies():
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "description": s["description"],
            "default_params": s["default_params"],
            "param_schema": s["param_schema"],
        }
        for s in STRATEGY_REGISTRY.values()
    ]


def get_strategy(strategy_id: str):
    return STRATEGY_REGISTRY.get(strategy_id)
