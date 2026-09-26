from typing import Any, Protocol, runtime_checkable

_PAYOUT = 1.0


@runtime_checkable
class SettlementModel(Protocol):
    """Turns a market's outcome into a payout.

    The broker calls this when a market closes, then flattens the position. A venue
    that settles to a scale rather than a winner replaces this.
    """

    def payout(self, result: Any, yes_contracts: float, no_contracts: float) -> float | None:
        """Cash the position pays out, or ``None`` if ``result`` is not recognised."""
        ...


class BinarySettlement:
    """Kalshi's rule: the winning side pays 1.0 a contract, the losing side nothing.

    An unrecognised result pays nothing and is reported by the broker rather than
    silently treated as a loss, since it means the market data is wrong.
    """

    def payout(self, result: Any, yes_contracts: float, no_contracts: float) -> float | None:
        if result == "yes":
            return yes_contracts * _PAYOUT
        if result == "no":
            return no_contracts * _PAYOUT
        return None
