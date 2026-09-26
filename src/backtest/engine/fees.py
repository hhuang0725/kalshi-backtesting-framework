_TAKER_RATE = 0.07
_MAKER_MULTIPLIER = 0.25


# Reference at-the-money strike. A fill's actual price must be supplied separately.
STRIKE_PRICE = 0.50


def taker_fee(p: float) -> float:
    return _TAKER_RATE * p * (1.0 - p)


def maker_fee(p: float) -> float:
    return _MAKER_MULTIPLIER * taker_fee(p)


def fee_for(order_type: str, p: float) -> float:
    return maker_fee(p) if order_type == "limit" else taker_fee(p)


def breakeven_for(entry_price: float) -> float:
    """Fee-adjusted breakeven win rate for a taker fill at ``entry_price``.

    ``entry_price`` is required because a reference strike is not necessarily the
    execution price.
    """
    return entry_price + taker_fee(entry_price)


# Reference breakeven at the at-the-money strike.
STRIKE_BREAKEVEN = breakeven_for(STRIKE_PRICE)
