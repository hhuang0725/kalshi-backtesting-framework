from typing import Protocol, runtime_checkable

from src.backtest.engine.fees import fee_for


@runtime_checkable
class FeeModel(Protocol):
    """Prices one fill without retaining order or broker state.

    Consulted on the estimated fill when an order is admitted, then on each real
    fill. Implementations may model a per-contract charge, an execution commission,
    or a rebate.
    """

    def charge(self, order_type: str, price: float, quantity: float) -> float:
        """Fee for ``quantity`` contracts filling at ``price``."""
        ...


class KalshiFees:
    """Kalshi's schedule: ``0.07 * p * (1 - p)`` per contract, a quarter of that for a maker.

    The rate is symmetric in ``p``, so a NO fill priced at ``1 - p`` is charged the
    same as the YES fill it mirrors.
    """

    def charge(self, order_type: str, price: float, quantity: float) -> float:
        return fee_for(order_type, price) * quantity
