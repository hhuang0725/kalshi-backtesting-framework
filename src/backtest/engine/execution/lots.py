from typing import Protocol, runtime_checkable


@runtime_checkable
class LotPolicy(Protocol):
    """Rounds a strategy's requested size to something the venue will accept.

    Applied when an order is admitted, before it can rest. A size that rounds to
    zero is refused rather than sent.
    """

    def normalize(self, quantity: float) -> float:
        """The tradeable size nearest ``quantity``, or ``0.0`` to refuse the order."""
        ...


class TenthLots:
    """Kalshi's lot size: contracts round to one decimal place.

    Kelly returns a fractional stake, so nearly every order is rounded here. Sizes
    below 0.05 round to zero and the broker refuses them.
    """

    def normalize(self, quantity: float) -> float:
        return round(quantity, 1)
