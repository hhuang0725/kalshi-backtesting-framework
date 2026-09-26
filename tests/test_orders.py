import pytest

from src.backtest.engine.fees import maker_fee, taker_fee
from src.backtest.engine.orders import Order


@pytest.mark.unit
def test_fees():
    assert taker_fee(0.9) == pytest.approx(0.07 * 0.9 * 0.1)
    assert maker_fee(0.9) == pytest.approx(0.25 * taker_fee(0.9))


@pytest.mark.unit
def test_fee_symmetric_in_p_and_complement():
    assert taker_fee(0.9) == pytest.approx(taker_fee(0.1))


@pytest.mark.unit
def test_order_defaults():
    o = Order("yes", 3)
    assert o.side == "yes"
    assert o.quantity == 3
    assert o.order_type == "market"
    assert o.limit_price is None
