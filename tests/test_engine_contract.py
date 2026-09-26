"""Contract-level characterization of the engine's run semantics.

These tests pin *behavior that has no single owning module* — the ordering between
feeds, broker and strategy, what a strategy can see, and when settlement happens.
They allow modules to move without silently changing a timing rule. Every
assertion describes current behavior, including explicitly pinned limitations.

Deliberately data-free: a two-market, four-candle synthetic fixture exercises the
same paths as `data/master/*.csv` in milliseconds, so these run in the unit tier.
"""

import pytest

from src.backtest.engine import Engine
from src.backtest.engine.analyzer import Analyzer
from src.backtest.engine.context import AccountSnapshot
from src.backtest.engine.execution import KalshiBroker
from src.backtest.engine.feed import MarketFeed
from src.backtest.engine.fees import taker_fee
from src.backtest.engine.orders import Order
from src.backtest.engine.strategy import Strategy
from tests.helpers import ts, write_feed

# Two 2-minute markets over four 1-minute candles. Market A opens at 00:00 and closes
# at 00:02; market B opens at 00:02 and closes at 00:04. The shared boundary instant is
# the point of MarketFeed's strict-open rule, so it is built into the fixture.
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


def _engine(tmp_path, starting_balance: float = 1000.0) -> Engine:
    """A minimal two-feed Kalshi-shaped engine over the fixture above."""
    candle = write_feed(tmp_path, "candle", _CANDLES)
    market = write_feed(tmp_path, "market", _MARKETS, cls=MarketFeed)
    engine = Engine()
    engine.add_data_feed(candle)        # first = master clock
    engine.add_data_feed(market)
    engine.broker = KalshiBroker(starting_balance, master_feed=candle, market_feed=market)
    return engine


class _Recorder(Strategy):
    """Records what the context showed on every tick, and optionally places one order."""

    def __init__(self, orders: dict[int, Order] | None = None) -> None:
        self.orders = orders or {}
        self.ticks: list[dict] = []

    def next(self, ctx) -> list[Order]:
        tick = len(self.ticks)
        self.ticks.append({
            "market_ticker": ctx.market.ticker[0] if len(ctx.market) else None,
            "candle_ticker": ctx.candle.ticker[0],
            "masked_result": ctx.market.result[0] if len(ctx.market) else None,
            "balance": ctx.balance,
            "yes": ctx.yes_contracts,
            "no": ctx.no_contracts,
        })
        order = self.orders.get(tick)
        return [order] if order else []


# --- feed clocking -----------------------------------------------------------------


@pytest.mark.unit
def test_master_steps_one_row_per_tick(tmp_path):
    """The first registered feed is the clock: exactly one bar per tick, size() ticks total."""
    engine = _engine(tmp_path)
    rec = _Recorder()
    engine.run([rec])
    assert [t["candle_ticker"] for t in rec.ticks] == ["A", "A", "B", "B"]


@pytest.mark.unit
def test_market_feed_holds_the_closing_market_through_its_final_candle(tmp_path):
    """Strict-open is why the 00:02 candle still belongs to market A, not B:
    consecutive markets share the boundary instant, and the candle closing there is
    A's last."""
    engine = _engine(tmp_path)
    rec = _Recorder()
    engine.run([rec])
    assert [t["market_ticker"] for t in rec.ticks] == ["A", "A", "B", "B"]


# --- lookahead protection ----------------------------------------------------------


@pytest.mark.unit
def test_active_market_outcome_is_masked_every_tick(tmp_path):
    """The current market's result is never visible to the strategy."""
    engine = _engine(tmp_path)
    rec = _Recorder()
    engine.run([rec])
    assert [t["masked_result"] for t in rec.ticks] == [None, None, None, None]


@pytest.mark.unit
def test_settled_market_outcome_is_visible_at_minus_one(tmp_path):
    """Masking is positional, not permanent: once a market is past, its result reads."""
    market = write_feed(tmp_path, "market", _MARKETS, cls=MarketFeed)
    market.advance()
    market.advance()
    assert market.result[0] is None         # B, still active
    assert market.result[-1] == "yes"       # A, settled
    assert market.expiration_value[-1] == pytest.approx(1.0)
    assert market.volume_fp[-1] == 10


@pytest.mark.unit
def test_positive_index_is_rejected(tmp_path):
    """The structural lookahead guard: there is no syntax for reading the future."""
    candle = write_feed(tmp_path, "candle", _CANDLES)
    candle.advance()
    with pytest.raises(IndexError, match="lookahead"):
        candle.yes_ask_open[1]


# --- order timing ------------------------------------------------------------------


@pytest.mark.unit
def test_order_fills_on_the_tick_after_it_is_submitted(tmp_path):
    """The one-candle delay, and the exact price it implies. `update()` runs before
    `next()`, so an order returned on tick 0 is only resting once that tick's fill
    pass is over: it fills on tick 1 at 0.60, never tick 0's 0.50."""
    engine = _engine(tmp_path)
    rec = _Recorder({0: Order("yes", 1, "market")})
    engine.run([rec])
    broker = engine.brokers[0]
    assert broker.orders_filled == 1
    # The context is built after update(), so tick 1's snapshot already reflects the fill.
    assert rec.ticks[0]["yes"] == 0.0
    assert rec.ticks[1]["yes"] == 1.0
    # One contract at tick 1's opening ask, plus the taker fee on that price.
    assert rec.ticks[1]["balance"] == pytest.approx(1000.0 - 0.60 - taker_fee(0.60))


@pytest.mark.unit
def test_resting_limit_is_cancelled_when_its_market_closes(tmp_path):
    """Unfilled limit orders do not survive settlement into the next market."""
    engine = _engine(tmp_path)
    # A limit at 0.10 can never fill against this fixture's quotes.
    rec = _Recorder({0: Order("yes", 1, "limit", 0.10)})
    engine.run([rec])
    broker = engine.brokers[0]
    assert broker.orders_submitted == 1
    assert broker.orders_filled == 0
    assert broker._resting == []


# --- settlement --------------------------------------------------------------------


@pytest.mark.unit
def test_market_settles_when_the_next_market_becomes_active(tmp_path):
    """A's YES position pays out on tick 2, the tick B becomes current."""
    engine = _engine(tmp_path)
    rec = _Recorder({0: Order("yes", 1, "market")})
    engine.run([rec])
    trades = engine.brokers[0].get_trades()
    assert len(trades) == 1
    assert trades[0]["won"] is True
    assert trades[0]["side"] == "yes"
    # Settled at tick 2, so the snapshot the strategy saw on tick 2 is already flat.
    assert rec.ticks[2]["yes"] == 0.0


@pytest.mark.unit
def test_final_market_is_never_settled(tmp_path):
    """The current implementation settles only when the market id changes."""
    engine = _engine(tmp_path)
    # Buy YES in market A (tick 0 -> fills tick 1) and NO in market B (tick 2 -> fills tick 3).
    rec = _Recorder({0: Order("yes", 1, "market"), 2: Order("no", 1, "market")})
    engine.run([rec])
    broker = engine.brokers[0]
    assert broker.orders_filled == 2            # both orders did fill
    assert len(broker.get_trades()) == 1        # but only market A produced a trade
    assert broker.no_contracts == 1.0           # B's position is still open at the end


# --- multi-strategy isolation ------------------------------------------------------


@pytest.mark.unit
def test_each_strategy_gets_an_independent_broker(tmp_path):
    """One feed traversal, one broker per strategy, no shared account state."""
    engine = _engine(tmp_path)
    buyer = _Recorder({0: Order("yes", 1, "market")})
    noop = _Recorder()
    engine.run([buyer, noop])

    assert len(engine.brokers) == 2
    assert engine.brokers[0] is not engine.brokers[1]
    assert engine.brokers[0].orders_filled == 1
    assert engine.brokers[1].orders_filled == 0
    # The noop strategy's view of its account is untouched by the buyer's fills.
    assert [t["balance"] for t in noop.ticks] == [1000.0] * 4


@pytest.mark.unit
def test_feeds_are_shared_across_brokers_not_copied(tmp_path):
    """Feeds advance once per tick for every strategy — the reason batching is cheap."""
    engine = _engine(tmp_path)
    a, b = _Recorder(), _Recorder()
    engine.run([a, b])
    assert [t["candle_ticker"] for t in a.ticks] == [t["candle_ticker"] for t in b.ticks]


@pytest.mark.unit
def test_rerun_resets_feeds_and_brokers(tmp_path):
    """The same engine and equivalent strategies produce identical results twice."""
    engine = _engine(tmp_path)
    first = engine.run([_Recorder({0: Order("yes", 1, "market")})])
    second = engine.run([_Recorder({0: Order("yes", 1, "market")})])
    assert first == second


# --- analyzers ---------------------------------------------------------------------


class _CountingAnalyzer(Analyzer):
    """Records when it ran and what the broker looked like at that moment."""

    def __init__(self, key: str, value) -> None:
        self.key = key
        self.value = value
        self.calls: list[int] = []

    def metrics(self, broker) -> dict:
        self.calls.append(len(broker.get_trades()))
        return {self.key: self.value}


@pytest.mark.unit
def test_analyzer_runs_once_per_broker_after_the_loop(tmp_path):
    engine = _engine(tmp_path)
    counter = _CountingAnalyzer("probe", 1)
    engine.add_analyzer(counter)
    engine.run([_Recorder({0: Order("yes", 1, "market")}), _Recorder()])
    # One call per strategy, and each saw a finished broker (strategy 0 has its trade).
    assert counter.calls == [1, 0]


@pytest.mark.unit
def test_later_analyzer_wins_on_key_collision(tmp_path):
    """Merge order is attach order, last write winning. Pinned because the refactor
    proposes detecting collisions instead — that would be a deliberate change."""
    engine = _engine(tmp_path)
    engine.add_analyzer(_CountingAnalyzer("win_rate", "first"))
    engine.add_analyzer(_CountingAnalyzer("win_rate", "second"))
    (metrics,) = engine.run([_Recorder()])
    assert metrics["win_rate"] == "second"


# --- broker configuration ----------------------------------------------------------


@pytest.mark.unit
def test_fork_shares_feeds_and_resets_state(tmp_path):
    """Forks retain broker wiring without carrying run state."""
    engine = _engine(tmp_path)
    template = engine.broker
    template._account.buy("yes", 10, 0.5)       # dirty the account and the record
    template.trades.append({"pnl": 1.0})

    twin = template.fork()
    assert not hasattr(template, "clone")
    assert twin.master_feed is template.master_feed
    assert twin.market_feed is template.market_feed
    assert twin.starting_balance == template.starting_balance
    assert twin.balance == template.starting_balance
    assert twin.yes_contracts == 0.0
    assert twin.trades == []
    # The account is copied, not shared, or resetting the fork would flatten both.
    assert twin._account is not template._account
    assert template.yes_contracts == 10.0


@pytest.mark.unit
def test_fork_carries_non_default_configuration(tmp_path):
    """Copying preserves constructor configuration without maintaining a second list."""
    engine = _engine(tmp_path)
    configured = KalshiBroker(500.0, master_feed=engine.broker.master_feed,
                        market_feed=engine.broker.market_feed,
                        id_col="ticker", result_col="result")
    twin = configured.fork()
    assert twin.starting_balance == 500.0
    assert twin.id_col == "ticker"
    assert twin.result_col == "result"
@pytest.mark.unit
def test_finalize_runs_once_per_broker_and_is_idempotent(tmp_path):
    """Finalization is currently an idempotent lifecycle hook."""
    engine = _engine(tmp_path)
    rec = _Recorder({2: Order("no", 1, "market")})
    engine.run([rec])
    broker = engine.brokers[0]
    assert broker._finalized is True
    before = (len(broker.get_trades()), broker.no_contracts, broker.balance)
    broker.finalize()
    assert (len(broker.get_trades()), broker.no_contracts, broker.balance) == before


@pytest.mark.unit
def test_the_account_attribute_never_goes_stale(tmp_path):
    """The risk the kept-current account trades for speed, pinned.

    A mutation path that forgot to report itself would serve a stale account, and a
    strategy would size against a balance it no longer has. This walks a run that
    settles and fills, checking the attribute against a fresh one every tick.
    """
    engine = _engine(tmp_path)
    broker = engine.broker.fork()
    market = engine._feeds[1]
    candle = engine._feeds[0]
    orders = {0: Order("yes", 1, "market"), 2: Order("no", 1, "market")}

    for tick in range(candle.size()):
        candle.step()
        market.advance_to(candle.current_time())
        broker.update()
        fresh = AccountSnapshot(cash=broker.balance,
                                positions={"yes": broker.yes_contracts, "no": broker.no_contracts})
        assert broker.account.cash == fresh.cash, f"stale cash at tick {tick}"
        assert broker.account.positions == fresh.positions, f"stale positions at tick {tick}"
        if tick in orders:
            broker.receive_orders([orders[tick]])

    assert broker.orders_filled == 2         # the run really did move the account


@pytest.mark.unit
def test_the_account_object_is_reused_while_nothing_moves(tmp_path):
    """Why it is an attribute and not a method: most ticks hand back the same object."""
    engine = _engine(tmp_path)
    broker = engine.broker.fork()
    before = broker.account
    broker.update()                          # no market visible yet, nothing settles
    assert broker.account is before


@pytest.mark.unit
def test_engine_requires_feeds_and_a_broker(tmp_path):
    empty = Engine()
    with pytest.raises(ValueError, match="no feeds"):
        empty.run([_Recorder()])

    candle = write_feed(tmp_path, "candle", _CANDLES)
    no_broker = Engine()
    no_broker.add_data_feed(candle)
    with pytest.raises(ValueError, match="broker"):
        no_broker.run([_Recorder()])
