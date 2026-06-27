from core.config import MAX_DAILY_LOSS_PCT, MAX_POSITION_SIZE, MAX_SHARES_PER_TRADE
from core.database import is_kill_switch_on, get_setting
from core.logger import get_logger

log = get_logger("risk_engine")

class RiskEngine:
    def __init__(self, alpaca_adapter):
        self.alpaca = alpaca_adapter

    def check(self, symbol: str, action: str, quantity: int) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Runs all safety checks before an order is placed.
        """

        # 1. Kill switch
        if is_kill_switch_on():
            return False, "Kill switch is ON — all trading halted"

        # 2. Trading enabled flag
        if get_setting("trading_enabled") != "1":
            return False, "Trading is disabled in settings"

        # 3. Quantity sanity
        if quantity <= 0:
            return False, f"Invalid quantity: {quantity}"
        if quantity > MAX_SHARES_PER_TRADE:
            return False, f"Quantity {quantity} exceeds max {MAX_SHARES_PER_TRADE} shares per trade"

        # 4. Max open positions
        try:
            positions = self.alpaca.get_positions()
            if action == "buy" and len(positions) >= MAX_POSITION_SIZE:
                return False, f"Max open positions ({MAX_POSITION_SIZE}) reached"
        except Exception as e:
            log.warning(f"Could not fetch positions for risk check: {e}")

        # 5. Daily loss limit
        try:
            account = self.alpaca.get_account()
            equity      = float(account.get("equity", 0))
            last_equity = float(account.get("last_equity", equity))
            if last_equity > 0:
                daily_loss_pct = ((last_equity - equity) / last_equity) * 100
                if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                    return False, f"Daily loss limit hit: -{daily_loss_pct:.2f}% (limit {MAX_DAILY_LOSS_PCT}%)"
        except Exception as e:
            log.warning(f"Could not fetch account for risk check: {e}")

        return True, "OK"
