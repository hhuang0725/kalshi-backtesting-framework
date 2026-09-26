from typing import Protocol, runtime_checkable

from src.backtest.engine.context import AccountSnapshot
from src.backtest.engine.orders import Order


@runtime_checkable
class Broker(Protocol):
    """Handles orders: whether they fill, at what price, what they cost, and what
    that does to the account.

    :meth:`receive_orders` takes the strategy's orders and holds them until
    :meth:`update` processes them on a later tick; ``account`` reports the result.
    Fill model, fee schedule, position accounting and settlement all live here, so
    the engine never sees them. Analyzers read the broker after :meth:`finalize`.
    """

    def fork(self) -> "Broker":
        """An independent broker with the same configuration and no carried state."""
        ...

    def update(self) -> None:
        """Process one tick: lifecycle transitions, then orders resting from earlier ticks."""
        ...

    def receive_orders(self, orders: list[Order]) -> list[bool]:
        """Accept or refuse each order. Accepted orders rest until a later tick."""
        ...

    account: AccountSnapshot
    """The account right now. An attribute the broker keeps current rather than a
    method, because the engine reads it once per tick per strategy."""

    def finalize(self) -> None:
        """Close the run. Called once after the last tick, before any analyzer."""
        ...
