"""The five policies KalshiBroker delegates to, and that they can be swapped.

test_broker.py covers the broker orchestrating them. This file covers each policy
on its own and verifies that different fee and fill models work without editing
the broker or engine.
"""

import pytest

from src.backtest.engine import Engine
from src.backtest.engine.execution.account import BinaryAccount
from src.backtest.engine.execution.fees import FeeModel, KalshiFees
from src.backtest.engine.execution.fills import CandleFills, FillModel
from src.backtest.engine.execution.lots import LotPolicy, TenthLots
from src.backtest.engine.execution.settlement import BinarySettlement, SettlementModel
from src.backtest.engine.execution.simulated_broker import KalshiBroker
from src.backtest.engine.fees import taker_fee
from src.backtest.engine.orders import Fill, Order
from src.backtest.engine.strategy import Strategy
from tests.helpers import ts, write_feed


def _quote_feed(tmp_path, **prices):
    row = {"close_time": ts(1), "yes_ask_open": 0.5, "yes_bid_open": 0.5,
           "yes_ask_low": 0.5, "yes_bid_high": 0.5}
    row.update(prices)
    feed = write_feed(tmp_path, "candle", [row])
    feed.advance()
    return feed


# --- fills -------------------------------------------------------------------------


@pytest.mark.unit
def test_market_orders_fill_at_the_opening_quote(tmp_path):
    fills = CandleFills()
    yes = fills.execute(Order("yes", 1, "market"), _quote_feed(tmp_path, yes_ask_open=0.62))
    no = fills.execute(Order("no", 1, "market"), _quote_feed(tmp_path, yes_bid_open=0.55))
    assert yes == Fill(price=pytest.approx(0.62), quantity=1)
    assert no == Fill(price=pytest.approx(0.55), quantity=1)


@pytest.mark.unit
def test_a_yes_limit_fills_only_if_the_bar_traded_through_it(tmp_path):
    fills = CandleFills()
    order = Order("yes", 1, "limit", 0.50)
    assert fills.execute(order, _quote_feed(tmp_path, yes_ask_low=0.48)) == Fill(0.50, 1)
    assert fills.execute(order, _quote_feed(tmp_path, yes_ask_low=0.52)) is None


@pytest.mark.unit
def test_a_no_limit_is_priced_off_the_yes_bid(tmp_path):
    """no_ask_low = 1 - yes_bid_high, and the fill is reported YES-denominated."""
    fills = CandleFills()
    order = Order("no", 1, "limit", 0.40)
    fill = fills.execute(order, _quote_feed(tmp_path, yes_bid_high=0.65))
    assert fill == Fill(price=pytest.approx(0.60), quantity=1)
    assert fills.execute(order, _quote_feed(tmp_path, yes_bid_high=0.55)) is None


@pytest.mark.unit
def test_a_limit_order_without_a_price_is_rejected(tmp_path):
    feed = _quote_feed(tmp_path)
    broker = KalshiBroker(100.0, feed, None)
    assert broker.receive_orders([Order("yes", 1, "limit", None)]) == [False]

    with pytest.raises(ValueError):
        CandleFills().execute(Order("yes", 1, "limit", None), feed)


@pytest.mark.unit
def test_admission_estimate_uses_the_fill_model_not_broker_quote_logic(tmp_path):
    feed = _quote_feed(tmp_path, yes_ask_open=0.90)
    broker = KalshiBroker(
        1.0,
        master_feed=feed,
        market_feed=None,
        fills=FreeFills(0.10),
        fees=FlatFee(0.0),
    )
    assert broker.receive_orders([Order("yes", 1, "market")]) == [True]


# --- fees, lots, settlement --------------------------------------------------------


@pytest.mark.unit
def test_kalshi_fees_scale_with_quantity():
    assert KalshiFees().charge("market", 0.4, 10) == pytest.approx(taker_fee(0.4) * 10)


@pytest.mark.unit
def test_maker_pays_a_quarter_of_taker():
    fees = KalshiFees()
    assert fees.charge("limit", 0.4, 1) == pytest.approx(fees.charge("market", 0.4, 1) * 0.25)


@pytest.mark.unit
def test_lots_round_to_a_tenth_and_sub_lot_sizes_vanish():
    lots = TenthLots()
    assert lots.normalize(1.24) == pytest.approx(1.2)
    assert lots.normalize(0.04) == 0.0


@pytest.mark.unit
def test_binary_settlement_pays_the_winning_side_only():
    s = BinarySettlement()
    assert s.payout("yes", 10.0, 0.0) == pytest.approx(10.0)
    assert s.payout("no", 10.0, 0.0) == pytest.approx(0.0)
    assert s.payout("", 10.0, 0.0) is None       # unrecognised, so the broker can warn


# --- account ------------------------------------------------------------------------


@pytest.mark.unit
def test_buying_the_opposite_side_closes_before_it_opens():
    account = BinaryAccount(1000.0)
    account.buy("yes", 3, 0.5)
    account.buy("no", 5, 0.5)
    assert account.yes_contracts == 0
    assert account.no_contracts == 2             # closed 3, opened 2
    assert min(account.yes_contracts, account.no_contracts) == 0


@pytest.mark.unit
def test_cost_to_buy_accounts_for_what_closing_releases():
    account = BinaryAccount(1000.0)
    account.buy("yes", 10, 0.5)
    # Closing 5 YES at 0.5 returns 2.5; nothing is opened, so the buy pays nothing net.
    assert account.cost_to_buy("no", 5, 0.5) == pytest.approx(-2.5)


@pytest.mark.unit
def test_the_balance_floor_is_what_makes_an_order_unaffordable():
    account = BinaryAccount(1.0)
    assert account.can_afford(0.5)
    assert not account.can_afford(0.51)


# --- the swap ----------------------------------------------------------------------


class FreeFills:
    """Every order fills at a fixed price, whatever the quotes say."""

    def __init__(self, price: float) -> None:
        self.price_ = price

    def estimate(self, order, feed):
        return Fill(self.price_, order.quantity)

    def execute(self, order, feed):
        return Fill(self.price_, order.quantity)


class OneAtATime(FreeFills):
    """Fills at most one contract from an order on each eligible tick."""

    def estimate(self, order, feed):
        return Fill(self.price_, min(1.0, order.quantity))

    def execute(self, order, feed):
        return Fill(self.price_, min(1.0, order.quantity))


class FlatFee:
    """A per-order commission instead of Kalshi's price-dependent rate."""

    def __init__(self, amount: float) -> None:
        self.amount = amount

    def charge(self, order_type, price, quantity):
        return self.amount


class WholeLots:
    def normalize(self, quantity: float) -> float:
        return float(int(quantity))


class BuyOnce(Strategy):
    def reset(self):
        self.done = False

    def next(self, ctx):
        if self.done or len(ctx.market) == 0:
            return []
        self.done = True
        return [Order("yes", 2.4, "market")]


# Two markets, so the first settles and leaves a trade record: the last market of a
# run is never settled (see test_engine_contract.test_final_market_is_never_settled).
_CANDLES = [
    {"close_time": ts(1), "ticker": "A", "yes_ask_open": 0.50, "yes_bid_open": 0.48,
     "yes_ask_low": 0.50, "yes_bid_high": 0.48},
    {"close_time": ts(2), "ticker": "A", "yes_ask_open": 0.60, "yes_bid_open": 0.58,
     "yes_ask_low": 0.60, "yes_bid_high": 0.58},
    {"close_time": ts(3), "ticker": "B", "yes_ask_open": 0.70, "yes_bid_open": 0.68,
     "yes_ask_low": 0.70, "yes_bid_high": 0.68},
    {"close_time": ts(4), "ticker": "B", "yes_ask_open": 0.80, "yes_bid_open": 0.78,
     "yes_ask_low": 0.80, "yes_bid_high": 0.78},
]
_MARKETS = [
    {"open_time": ts(0), "close_time": ts(2), "ticker": "A", "result": "yes",
     "expiration_value": 1.0, "volume_fp": 10},
    {"open_time": ts(2), "close_time": ts(4), "ticker": "B", "result": "no",
     "expiration_value": 0.0, "volume_fp": 20},
]


def _run(tmp_path, **policies):
    from src.backtest.engine.feeds.kalshi_market import MarketFeed

    tmp_path.mkdir(parents=True, exist_ok=True)
    candle = write_feed(tmp_path, "candle", _CANDLES)
    market = write_feed(tmp_path, "market", _MARKETS, cls=MarketFeed)
    engine = Engine()
    engine.add_data_feed(candle)
    engine.add_data_feed(market)
    engine.broker = KalshiBroker(1000.0, master_feed=candle, market_feed=market, **policies)
    engine.run([BuyOnce()])
    return engine.brokers[0]


@pytest.mark.unit
def test_a_different_fill_model_needs_no_broker_or_engine_edit(tmp_path):
    default = _run(tmp_path / "a")
    swapped = _run(tmp_path / "b", fills=FreeFills(0.10))
    assert default.orders_filled == swapped.orders_filled == 1
    # The cheaper fill leaves more cash, which is the whole observable difference.
    assert swapped.balance > default.balance


@pytest.mark.unit
def test_a_different_fee_model_needs_no_broker_or_engine_edit(tmp_path):
    default = _run(tmp_path / "a")
    swapped = _run(tmp_path / "b", fees=FlatFee(1.0))
    assert swapped.orders_filled == 1
    assert default.balance != swapped.balance


@pytest.mark.unit
def test_a_different_lot_policy_changes_the_size_that_is_admitted(tmp_path):
    """The strategy asks for 2.4; tenths keep it, whole lots round it to 2."""
    assert _run(tmp_path / "a").get_trades()[0]["contracts"] == pytest.approx(2.4)
    assert _run(tmp_path / "b", lots=WholeLots()).get_trades()[0]["contracts"] == pytest.approx(2.0)


@pytest.mark.unit
def test_a_partial_fill_leaves_only_the_remainder_resting(tmp_path):
    feed = _quote_feed(tmp_path)
    broker = KalshiBroker(
        100.0,
        master_feed=feed,
        market_feed=None,
        fills=OneAtATime(0.40),
        fees=FlatFee(0.0),
    )
    assert broker.receive_orders([Order("yes", 2.4, "market")]) == [True]

    assert broker._fill_resting()
    assert broker.yes_contracts == pytest.approx(1.0)
    assert broker._resting[0].order.quantity == pytest.approx(1.4)
    assert broker.orders_filled == 1

    assert broker._fill_resting()
    assert broker._fill_resting()
    assert broker.yes_contracts == pytest.approx(2.4)
    assert broker._resting == []
    assert broker.orders_filled == 1


@pytest.mark.unit
def test_trade_prices_are_weighted_by_partial_fill_quantity():
    broker = KalshiBroker(100.0, None, None, fees=FlatFee(0.0))
    broker._apply_fill(Order("yes", 1.0, "market"), 0.20)
    broker._apply_fill(Order("yes", 2.0, "market"), 0.40)
    broker._settle("yes", ts(1))

    trade = broker.get_trades()[0]
    assert trade["contracts"] == pytest.approx(3.0)
    assert trade["entry_price"] == pytest.approx((0.20 + 2 * 0.40) / 3)
    assert trade["paid"] == pytest.approx((0.20 + 2 * 0.40) / 3)


@pytest.mark.unit
def test_a_fill_model_cannot_execute_more_than_the_order(tmp_path):
    class Overfill(FreeFills):
        def execute(self, order, feed):
            return Fill(self.price_, order.quantity + 1.0)

    feed = _quote_feed(tmp_path)
    broker = KalshiBroker(100.0, feed, None, fills=Overfill(0.40))
    broker.receive_orders([Order("yes", 1, "market")])
    with pytest.raises(ValueError, match="fill quantity"):
        broker._fill_resting()


@pytest.mark.unit
def test_the_policies_satisfy_their_protocols():
    assert isinstance(KalshiFees(), FeeModel)
    assert isinstance(CandleFills(), FillModel)
    assert isinstance(TenthLots(), LotPolicy)
    assert isinstance(BinarySettlement(), SettlementModel)
    assert isinstance(FlatFee(1.0), FeeModel)
    assert isinstance(FreeFills(0.1), FillModel)
    assert isinstance(WholeLots(), LotPolicy)
