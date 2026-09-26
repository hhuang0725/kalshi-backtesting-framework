import logging

import numpy as np

from src.backtest.engine.context import Context
from src.backtest.engine.optimization.parameters import Continuous
from src.backtest.engine.orders import Order
from src.backtest.engine.strategy import Strategy

logger = logging.getLogger(__name__)

_ONE_MINUTE = np.timedelta64(1, "m")

# Tunable parameters for src/backtest/engine/optimization/optimize.py (the strategy owns its space).
SEARCH_SPACE = {"threshold": Continuous(0.80, 0.98)}


class HighProbabilityStrategy(Strategy):
    """Enter when one side of the market first touches a high-confidence threshold.

    Each 1-minute candle of the active market is checked once: if the Yes side
    traded at or above ``threshold`` (``yes_bid_high``) a YES maker limit at
    ``threshold`` is submitted; if the No side did (``yes_ask_low <= 1 - threshold``,
    since ``no_bid = 1 - yes_ask``) a NO limit is submitted. Yes is checked first
    when both cross in the same candle. One entry per market; the broker cancels
    an unfilled limit at settlement.

    Every signal is appended to :attr:`signals` with the 1-indexed ``minute`` of
    the crossing and a ``filled`` flag observed from the broker position on later
    ticks — the raw material for the threshold/fill-rate/timing analysis.

    Lookahead invariant: only the just-closed candle ``ctx.candle[0]`` is read;
    the active market's masked ``result[0]`` is never an input.
    """

    def __init__(self, threshold: float = 0.90, quantity: int = 1) -> None:
        if not 0.5 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0.5, 1.0), got {threshold}")
        self.threshold = threshold
        # 1.0 - threshold can land one ulp below the decimal value (1.0 - 0.90 <
        # 0.10), which would miss exact touches; decimal-round it back onto the grid.
        self._no_trigger = round(1.0 - threshold, 10)
        self.quantity = quantity
        self.signals: list[dict] = []
        self._signaled_ticker = None

    def reset(self) -> None:
        """Drop the recorded signals and the entered-market flag for a fresh run.

        ``signals`` is the analysis output of one run, so a reused instance must not
        append a second run's signals onto the first's.
        """
        self.signals = []
        self._signaled_ticker = None

    def next(self, ctx: Context) -> list[Order]:
        if len(ctx.market) == 0 or len(ctx.candle) == 0:
            return []
        ticker = ctx.market.ticker[0]
        if ctx.candle.ticker[0] != ticker:      # candle/market feeds out of sync
            return []
        if ticker == self._signaled_ticker:     # already entered this market
            self._observe_fill(ctx)
            return []

        side = self._signal(ctx.candle)
        if side is None:
            return []
        self._signaled_ticker = ticker
        minute = self._minute(ctx)
        self.signals.append({
            "ticker": ticker,
            "side": side,
            "minute": minute,
            "entry_price": self.threshold,
            "filled": False,
        })
        logger.debug("[high_probability] %s -> %s at minute %d", ticker, side, minute)
        return [Order(side, self.quantity, order_type="limit", limit_price=self.threshold)]

    def _signal(self, candle) -> str | None:
        if float(candle.yes_bid_high[0]) >= self.threshold:
            return "yes"
        if float(candle.yes_ask_low[0]) <= self._no_trigger:
            return "no"
        return None

    def _minute(self, ctx: Context) -> int:
        """1-indexed minute of the signal candle within the 15m market window."""
        return int((ctx.candle.close_time[0] - ctx.market.open_time[0]) / _ONE_MINUTE)

    def _observe_fill(self, ctx: Context) -> None:
        """Mark the open signal filled once the broker shows a position on its side.

        Positions are zeroed at settlement and the strategy submits at most one
        order per market, so any contracts held while the signaled market is
        active belong to that order.
        """
        entry = self.signals[-1]
        if entry["filled"]:
            return
        held = ctx.yes_contracts if entry["side"] == "yes" else ctx.no_contracts
        if held > 0:
            entry["filled"] = True
