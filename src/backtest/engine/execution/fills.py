from typing import Protocol, runtime_checkable

from src.backtest.engine.feed import Feed
from src.backtest.engine.orders import Fill, Order


@runtime_checkable
class FillModel(Protocol):
    """Connects orders to execution data.

    Admission uses :meth:`estimate`; eligible resting orders use :meth:`execute`.
    The broker owns affordability, accounting and any unfilled remainder.
    """

    def estimate(self, order: Order, feed: Feed) -> Fill | None:
        """Expected fill for admission, or ``None`` when it cannot be estimated."""
        ...

    def execute(self, order: Order, feed: Feed) -> Fill | None:
        """Fill available now, or ``None`` when nothing executes."""
        ...


class CandleFills:
    """Fills against one candle of quotes: market orders at the open, limits if the
    bar traded through them.

    Prices are YES-denominated whichever side is bought, so the broker's cash math
    stays uniform in ``p``; a NO order's own cost is ``1 - p``. NO quotes are
    derived from YES, since ``no_ask = 1 - yes_bid`` and ``no_bid = 1 - yes_ask``.

    Size does not move the price here. A depth-aware model can replace that
    assumption without changing the broker.
    """

    def estimate(self, order: Order, feed: Feed) -> Fill | None:
        if len(feed) == 0:
            return None
        return Fill(self._reference_price(order, feed), order.quantity)

    def execute(self, order: Order, feed: Feed) -> Fill | None:
        if order.order_type == "market":
            return Fill(self._reference_price(order, feed), order.quantity)

        if order.limit_price is None:
            raise ValueError("limit order requires a limit_price")
        limit = order.limit_price

        # A resting maker order fills at its own price, only if the quote reached it.
        if order.side == "yes":
            if float(feed.yes_ask_low[0]) <= limit:
                return Fill(limit, order.quantity)
            return None
        # NO limit at `limit`: no_ask_low = 1 - yes_bid_high; fills when that <= limit.
        if (1.0 - float(feed.yes_bid_high[0])) <= limit:
            return Fill(1.0 - limit, order.quantity)
        return None

    @staticmethod
    def _reference_price(order: Order, feed: Feed) -> float:
        if order.order_type == "market":
            if order.side == "yes":
                return float(feed.yes_ask_open[0])
            return float(feed.yes_bid_open[0])      # no_ask_open = 1 - yes_bid_open
        if order.limit_price is None:
            raise ValueError("limit order requires a limit_price")
        return order.limit_price if order.side == "yes" else 1.0 - order.limit_price
