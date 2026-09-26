import pandas as pd
import pytest

from src.backtest.engine.execution import KalshiBroker
from src.backtest.engine.fees import taker_fee
from src.backtest.engine.orders import Order

_CLOSE = pd.Timestamp("2026-01-01T00:15:00+00:00")


def _broker() -> KalshiBroker:
    # feeds aren't needed for the fill/settlement math, which is exercised directly.
    b = KalshiBroker(1000.0, master_feed=None, market_feed=None)
    b._open_market()
    return b


# --- fill cash math + netting ---

@pytest.mark.unit
def test_buy_yes_market_costs_price_plus_fee():
    b = _broker()
    b._apply_fill(Order("yes", 10, "market"), 0.5)
    assert b.yes_contracts == 10
    assert b.no_contracts == 0
    assert b.balance == pytest.approx(1000.0 - 10 * 0.5 - taker_fee(0.5) * 10)


@pytest.mark.unit
def test_buy_no_while_long_yes_closes_and_returns_cash():
    b = _broker()
    b._apply_fill(Order("yes", 10, "market"), 0.5)
    bal = b.balance
    b._apply_fill(Order("no", 5, "market"), 0.5)
    assert b.yes_contracts == 5
    assert b.no_contracts == 0
    assert b.balance == pytest.approx(bal + 5 * 0.5 - taker_fee(0.5) * 5)


@pytest.mark.unit
def test_flip_in_one_order():
    b = _broker()
    b._apply_fill(Order("yes", 3, "market"), 0.5)
    b._apply_fill(Order("no", 5, "market"), 0.5)
    assert b.yes_contracts == 0
    assert b.no_contracts == 2          # closed 3, opened 2 NO


@pytest.mark.unit
def test_netting_invariant_holds():
    b = _broker()
    b._apply_fill(Order("yes", 4, "market"), 0.5)
    b._apply_fill(Order("no", 9, "market"), 0.5)
    assert min(b.yes_contracts, b.no_contracts) == 0


# --- settlement ---

@pytest.mark.unit
def test_settlement_pays_winning_side_and_records_trade():
    b = _broker()
    b._apply_fill(Order("yes", 10, "market"), 0.4)
    bal = b.balance
    b._settle("yes", _CLOSE)
    assert b.balance == pytest.approx(bal + 10)      # $1 per winning contract
    assert b.yes_contracts == 0
    assert len(b.trades) == 1
    assert b.trades[0]["won"] is True


@pytest.mark.unit
def test_settlement_losing_side_pays_nothing():
    b = _broker()
    b._apply_fill(Order("yes", 10, "market"), 0.4)
    bal = b.balance
    b._settle("no", _CLOSE)
    assert b.balance == pytest.approx(bal)
    assert b.trades[0]["won"] is False


@pytest.mark.unit
def test_reset_clears_state():
    b = _broker()
    b._apply_fill(Order("yes", 10, "market"), 0.5)
    b.reset()
    assert b.balance == 1000.0
    assert b.yes_contracts == 0
    assert b.trades == []
    assert b._last_id is None
