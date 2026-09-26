from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    side: str                       # "yes" | "no"
    quantity: float
    order_type: str = "market"      # "limit" | "market"
    limit_price: float | None = None    # in the order's own side terms


@dataclass(frozen=True)
class Fill:
    """The price and quantity an execution model can fill now."""

    price: float
    quantity: float
