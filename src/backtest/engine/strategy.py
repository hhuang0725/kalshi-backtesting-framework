from src.backtest.engine.context import Context
from src.backtest.engine.orders import Order


class Strategy:
    """The model that makes decisions during backtesting.

    :meth:`next` receives a :class:`Context` describing the current state, and
    returns a list of orders, which can be empty. It never moves money or data
    itself; the broker does that.
    """

    def reset(self) -> None:
        """Clear per-run state. Called by the engine before the first tick.

        A stateful strategy must override this or it will retain previous information.
        """

    def next(self, ctx: Context) -> list[Order]:
        raise NotImplementedError
