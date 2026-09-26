from src.backtest.engine.fees import fee_for


def kelly_fstar(win_rate: float, kelly_fraction: float, price: float = 0.50, order_type: str = "market") -> float:
    entry_cost = price + fee_for(order_type, price)
    b = (1.0 - entry_cost) / entry_cost
    return kelly_fraction * (win_rate - (1.0 - win_rate) / b)


def kelly_sizing(
    win_rate: float,
    kelly_fraction: float,
    balance: float,
    price: float,
    order_type: str = "market",
) -> float:
    """Kelly stake in (fractional) contracts. ``0.0`` when there is no edge.

    Returns the raw ``fstar * balance / entry_cost`` — **no minimum-contract floor
    and no integer truncation** (mirrors the live C++ ``market_kelly_size``). A
    non-positive ``fstar`` (win rate at/below the fee-adjusted breakeven) means no
    positive-EV bet, so the size is ``0.0`` and the trade is skipped. Rounding to the
    0.1-contract lot and dropping sub-lot sizes is the broker's job.
    """
    fstar = kelly_fstar(win_rate, kelly_fraction, price, order_type)
    if fstar <= 0.0:
        return 0.0
    entry_cost = price + fee_for(order_type, price)
    return fstar * balance / entry_cost
