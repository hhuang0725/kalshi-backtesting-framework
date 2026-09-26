"""A complete non-Kalshi stack driven by the same `Engine`.

Nothing in this file mentions YES, NO, contracts, tickers, markets,
settlement or CSV files, and none of the four components inherits from a first-party
class. If the engine ever grows a hidden dependency on the Kalshi shape, this file
stops running.

The venue modelled here is deliberately unlike Kalshi: a share that trades at a
quoted price, held indefinitely, with a flat per-trade commission and no settlement
event at all. Positions are fractional and long-only, and the account's instrument is
called ``"shares"``.
"""

import pytest

from src.backtest.engine import Engine
from src.backtest.engine.broker import Broker
from src.backtest.engine.context import AccountSnapshot
from src.backtest.engine.feed import Feed
from src.backtest.engine.orders import Order
from src.backtest.engine.strategy import Strategy

_PRICES = [10.0, 11.0, 12.0, 11.5, 13.0]
_COMMISSION = 0.05


class PriceView:
    """The only price-feed capabilities strategies receive."""

    __slots__ = ("__feed",)

    def __init__(self, feed: "PriceFeed") -> None:
        object.__setattr__(self, "_PriceView__feed", feed)

    def __getattribute__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @property
    def name(self) -> str:
        feed = object.__getattribute__(self, "_PriceView__feed")
        return feed.name

    def __len__(self) -> int:
        feed = object.__getattribute__(self, "_PriceView__feed")
        return len(feed)

    def price(self, ago: int = 0) -> float:
        feed = object.__getattribute__(self, "_PriceView__feed")
        return feed.price(ago)


class PriceFeed:
    """An in-memory feed. Implements `Feed` structurally, inherits nothing."""

    def __init__(self, name: str, prices: list[float]) -> None:
        self.name = name
        self._prices = list(prices)
        self._n = 0
        self._view = PriceView(self)

    def strategy_view(self) -> PriceView:
        return self._view

    def reset(self) -> None:
        self._n = 0

    def size(self) -> int:
        return len(self._prices)

    def step(self) -> None:
        self._n += 1

    def advance_to(self, master_time) -> None:
        while self._n < len(self._prices) and self._n <= master_time:
            self._n += 1

    def current_time(self):
        return self._n - 1

    def __len__(self) -> int:
        return self._n

    def price(self, ago: int = 0) -> float:
        """Column access on this feed's own terms — the engine never mediates it."""
        if ago > 0:
            raise IndexError("lookahead")
        return self._prices[self._n - 1 + ago]


class ShareBroker:
    """A cash-and-shares broker with a flat commission. Implements `Broker`."""

    def __init__(self, cash: float, feed: PriceFeed, commission: float = _COMMISSION) -> None:
        self.starting_cash = cash
        self.feed = feed
        self.commission = commission
        self.cash = cash
        self.shares = 0.0
        self.fills: list[tuple[float, float]] = []      # (quantity, price)
        self._resting: list[Order] = []
        self.finalized = 0

    def fork(self) -> "ShareBroker":
        return ShareBroker(self.starting_cash, self.feed, self.commission)

    def update(self) -> None:
        """Fill everything resting at this bar's price — the one-tick delay again."""
        for order in self._resting:
            price = self.feed.price(0)
            cost = order.quantity * price + self.commission
            if cost > self.cash:
                continue
            self.cash -= cost
            self.shares += order.quantity
            self.fills.append((order.quantity, price))
        self._resting = []

    def receive_orders(self, orders: list[Order]) -> list[bool]:
        self._resting.extend(orders)
        return [True] * len(orders)

    @property
    def account(self) -> AccountSnapshot:
        return AccountSnapshot(cash=self.cash, positions={"shares": self.shares})

    def finalize(self) -> None:
        self.finalized += 1

    def equity(self) -> float:
        return self.cash + self.shares * self.feed.price(0)


class BuyOnce(Strategy):
    """Buys one share the first time the price is below ``limit``."""

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.bought = False

    def reset(self) -> None:
        self.bought = False

    def next(self, ctx) -> list[Order]:
        if self.bought or ctx.prices.price(0) >= self.limit:
            return []
        self.bought = True
        return [Order("shares", 1.0)]


class EquityAnalyzer:
    """Reads a finished ShareBroker. Knows this broker's report, not the engine's."""

    def metrics(self, broker: ShareBroker) -> dict:
        return {"equity": broker.equity(), "n_fills": len(broker.fills),
                "shares": broker.shares, "finalized": broker.finalized}


def _engine(cash: float = 100.0) -> tuple[Engine, PriceFeed]:
    feed = PriceFeed("prices", _PRICES)
    engine = Engine()
    engine.add_data_feed(feed)
    engine.broker_factory = lambda: ShareBroker(cash, feed)
    engine.add_analyzer(EquityAnalyzer())
    return engine, feed


@pytest.mark.unit
def test_engine_runs_a_stack_with_no_kalshi_concepts():
    engine, _ = _engine()
    (metrics,) = engine.run([BuyOnce(limit=10.5)])
    # Buys on tick 0 at 10.0, fills on tick 1 at 11.0, then holds to the 13.0 close.
    assert metrics["n_fills"] == 1
    assert metrics["shares"] == pytest.approx(1.0)
    assert metrics["equity"] == pytest.approx(100.0 - 11.0 - _COMMISSION + 13.0)


@pytest.mark.unit
def test_broker_factory_gives_each_strategy_its_own_broker():
    engine, _ = _engine()
    buyer, sitter = BuyOnce(limit=10.5), BuyOnce(limit=0.0)
    bought, sat = engine.run([buyer, sitter])
    assert engine.brokers[0] is not engine.brokers[1]
    assert bought["n_fills"] == 1
    assert sat["n_fills"] == 0
    assert sat["equity"] == pytest.approx(100.0)


@pytest.mark.unit
def test_every_broker_is_finalized_exactly_once_before_analyzers():
    engine, _ = _engine()
    engine.run([BuyOnce(limit=10.5), BuyOnce(limit=99.0)])
    # The analyzer saw the count, so finalize() ran before metrics() were collected.
    assert [b.finalized for b in engine.brokers] == [1, 1]


@pytest.mark.unit
def test_strategy_state_is_reset_between_runs():
    """A reused instance gives the same answer twice because the engine resets it."""
    engine, _ = _engine()
    strategy = BuyOnce(limit=10.5)
    first = engine.run([strategy])
    second = engine.run([strategy])      # same object, already `bought` at the end of run 1
    assert first == second


@pytest.mark.unit
def test_account_snapshot_is_immutable_and_venue_neutral():
    engine, _ = _engine()

    seen: list[AccountSnapshot] = []

    class Peek(Strategy):
        def next(self, ctx) -> list[Order]:
            seen.append(ctx.account)
            return []

    engine.run([Peek()])
    assert all(isinstance(s, AccountSnapshot) for s in seen)
    assert seen[0].position("shares") == 0.0
    assert seen[0].position("anything-else") == 0.0      # absent instruments read as flat
    with pytest.raises((AttributeError, TypeError)):
        seen[0].cash = 1.0                               # frozen


@pytest.mark.unit
def test_a_strategy_cannot_write_through_its_account_snapshot():
    """`frozen` blocks rebinding `positions`, not editing the dict behind it, and
    the account object is reused across ticks — so one write would persist."""
    snapshot = AccountSnapshot(cash=100.0, positions={"shares": 1.0})
    for attempt in ('snapshot.positions["shares"] = 999.0',
                    'snapshot.positions["injected"] = 1.0',
                    'del snapshot.positions["shares"]'):
        with pytest.raises((TypeError, AttributeError)):
            exec(attempt)
    assert snapshot.position("shares") == 1.0


@pytest.mark.unit
def test_the_snapshot_does_not_alias_the_dict_it_was_built_from():
    """Wrapping without copying would leave it mutable through the caller's reference."""
    live = {"shares": 1.0}
    snapshot = AccountSnapshot(cash=100.0, positions=live)
    live["shares"] = 42.0
    assert snapshot.position("shares") == 1.0


@pytest.mark.unit
def test_context_exposes_the_master_timestamp():
    engine, _ = _engine()
    times: list = []

    class Clock(Strategy):
        def next(self, ctx) -> list[Order]:
            times.append(ctx.time)
            return []

    engine.run([Clock()])
    assert times == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_protocols_accept_the_fakes_structurally():
    """The fakes inherit nothing, so passing these checks is the point."""
    feed = PriceFeed("prices", _PRICES)
    assert isinstance(feed, Feed)
    assert isinstance(ShareBroker(100.0, feed), Broker)


@pytest.mark.unit
def test_duplicate_feed_names_are_rejected():
    engine = Engine()
    engine.add_data_feed(PriceFeed("prices", _PRICES))
    with pytest.raises(ValueError, match="duplicate feed name"):
        engine.add_data_feed(PriceFeed("prices", _PRICES))


@pytest.mark.unit
def test_strategies_cannot_advance_the_shared_feed():
    engine, _ = _engine()
    attempted: list[str] = []
    observed: list[float] = []

    class TriesToAdvance(Strategy):
        def next(self, ctx) -> list[Order]:
            for method in ("step", "advance", "advance_to", "reset"):
                with pytest.raises(AttributeError):
                    getattr(ctx.prices, method)
                attempted.append(method)
            return []

    class Observes(Strategy):
        def next(self, ctx) -> list[Order]:
            observed.append(ctx.prices.price())
            return []

    engine.run([TriesToAdvance(), Observes()])

    assert attempted == ["step", "advance", "advance_to", "reset"] * len(_PRICES)
    assert observed == _PRICES


@pytest.mark.unit
def test_strategy_view_does_not_expose_the_feed_or_future_prices():
    engine, _ = _engine()

    class ProbesInternals(Strategy):
        def next(self, ctx) -> list[Order]:
            for name in ("_feed", "_source", "_prices", "_n"):
                with pytest.raises(AttributeError):
                    getattr(ctx.prices, name)
            return []

    engine.run([ProbesInternals()])
