from pathlib import Path

import pandas as pd
import pytest

from src.backtest.engine import Engine, build_engine
from src.backtest.engine.analyzer import Analyzer
from src.backtest.engine.orders import Order
from src.backtest.engine.strategy import Strategy
from tests.helpers import ts, write_feed

_DATA = Path("data/master")
_HAS_DATA = (_DATA / "candlesticks.csv").exists()

pytestmark = pytest.mark.integration

_KEYS = {
    "markets_traded", "total_fills", "win_rate", "net_return", "net_return_pct", "avg_pnl",
    "flat_stake_ev", "max_drawdown", "max_drawdown_pct", "calmar", "min_equity",
    "fill_rate", "avg_entry_price",
}


class _TradeCount(Analyzer):
    """Dummy analyzer: one extra key, derived from the broker like a real one."""

    def metrics(self, broker) -> dict:
        return {"trade_count": len(broker.get_trades())}


class _Noop(Strategy):
    def next(self, ctx) -> list[Order]:
        return []


class _Probe(Strategy):
    def __init__(self) -> None:
        self.leaks = 0

    def next(self, ctx) -> list[Order]:
        if ctx.market.result[0] is not None:        # current outcome must never be visible
            self.leaks += 1
        return []


class _BuyYesMarket(Strategy):
    """Buy 1 YES market order on the first candle of each market."""

    def next(self, ctx) -> list[Order]:
        c = ctx.candle
        first_of_market = len(c) == 1 or c.ticker[-1] != c.ticker[0]
        if first_of_market and ctx.yes_contracts == 0 and ctx.no_contracts == 0:
            return [Order("yes", 1, "market")]
        return []


class _BuyYesLimit(Strategy):
    """Rest a YES limit at the bid on each market's first candle (maker path)."""

    def next(self, ctx) -> list[Order]:
        c = ctx.candle
        first_of_market = len(c) == 1 or c.ticker[-1] != c.ticker[0]
        if first_of_market and ctx.yes_contracts == 0 and ctx.no_contracts == 0:
            return [Order("yes", 1, "limit", 0.5)]
        return []


class _BuyCorrect(Strategy):
    """Oracle: buy the winning side (market order) on each market's first candle.

    The result is masked on ``ctx.market``, so the truth is supplied externally
    from ``markets.csv`` — a deliberate cheat for an upper-bound sanity check.
    """

    def __init__(self, oracle: dict) -> None:
        self.oracle = oracle

    def next(self, ctx) -> list[Order]:
        c = ctx.candle
        first_of_market = len(c) == 1 or c.ticker[-1] != c.ticker[0]
        if first_of_market and ctx.yes_contracts == 0 and ctx.no_contracts == 0:
            result = self.oracle.get(c.ticker[0])
            if result in ("yes", "no"):
                return [Order(result, 1, "market")]
        return []


class _Perfect(Strategy):
    """Future-seeing trader that rides every candle-to-candle move.

    With foresight of the whole price path it holds the side that profits from the
    next open-to-open move, flipping (1 contract) each candle and entering via a
    limit at the next candle's low — the best fill it could possibly get, at the
    maker fee. ``plan`` maps ``(ticker, idx)`` to ``(side, limit)``, precomputed
    from the full candlestick data.
    """

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self._ticker = None
        self._idx = 0

    def next(self, ctx) -> list[Order]:
        ticker = ctx.candle.ticker[0]
        if ticker != self._ticker:
            self._ticker, self._idx = ticker, 0
        else:
            self._idx += 1
        target = self.plan.get((ticker, self._idx))
        if target is None:
            return []
        side, limit = target
        if side == "yes" and ctx.yes_contracts == 0:
            return [Order("yes", 1 + ctx.no_contracts, "limit", limit)]    # close any NO, open 1 YES
        if side == "no" and ctx.no_contracts == 0:
            return [Order("no", 1 + ctx.yes_contracts, "limit", limit)]    # close any YES, open 1 NO
        return []


@pytest.fixture(scope="module")
def eng() -> Engine:
    if not _HAS_DATA:
        pytest.skip("data/master/*.csv not present")
    return build_engine()


# --- generic engine wiring (no data needed) ---

@pytest.mark.unit
def test_add_data_feed_preserves_order(tmp_path):
    e = Engine()
    a = write_feed(tmp_path, "a", [{"close_time": ts(1), "v": 1}])
    b = write_feed(tmp_path, "b", [{"close_time": ts(1), "v": 1}])
    e.add_data_feed(a)      # first = master
    e.add_data_feed(b)
    assert [f.name for f in e._feeds] == ["a", "b"]


@pytest.mark.unit
def test_run_requires_feeds_and_broker():
    with pytest.raises(ValueError):
        Engine().run([_Noop()])


@pytest.mark.unit
def test_generic_engine_starts_without_analyzers():
    e = Engine()
    assert e.analyzers == []
    e.add_analyzer(_TradeCount())
    assert [type(a) for a in e.analyzers] == [_TradeCount]


# --- end-to-end via the Kalshi factory ---

def test_noop_runs_and_trades_nothing(eng: Engine):
    (metrics,) = eng.run([_Noop()])
    assert set(metrics) == _KEYS
    assert metrics["markets_traded"] == 0
    assert metrics["total_fills"] == 0


def test_buy_first_fills_every_market(eng: Engine):
    (metrics,) = eng.run([_BuyYesMarket()])
    assert metrics["markets_traded"] > 0
    assert metrics["fill_rate"] == pytest.approx(1.0)


def test_no_current_result_leak(eng: Engine):
    probe = _Probe()
    eng.run([probe])
    assert probe.leaks == 0


def test_rerun_is_deterministic(eng: Engine):
    m1 = eng.run([_BuyYesMarket()])
    m2 = eng.run([_BuyYesMarket()])
    assert m1 == m2         # feeds + brokers reset cleanly between runs


def test_added_analyzer_merges_into_results(eng: Engine):
    eng.add_analyzer(_TradeCount())
    try:
        (m,) = eng.run([_BuyYesMarket()])
    finally:
        eng.analyzers.pop()         # eng is module-scoped; leave it as we found it
    assert set(m) == _KEYS | {"trade_count"}
    assert m["trade_count"] == m["markets_traded"]


def test_multiple_strategies_run_independently(eng: Engine):
    buy_m, noop_m = eng.run([_BuyYesMarket(), _Noop()])
    assert buy_m["markets_traded"] > 0
    assert noop_m["markets_traded"] == 0
    # the multi-run result matches running BuyFirst on its own (no cross-talk)
    (alone,) = eng.run([_BuyYesMarket()])
    assert buy_m == alone


# --- oracle / perfect upper-bound checks (validate settlement + fills end-to-end) ---

def _oracle() -> dict:
    """Map each market ticker to its true result, straight from the master data."""
    markets = pd.read_csv(_DATA / "markets.csv")
    return dict(zip(markets["ticker"], markets["result"]))


def _perfect_plan() -> dict:
    """Per-(ticker, candle-index) ``(side, limit)`` for the future-seeing trader.

    An order placed at candle ``idx`` fills at candle ``idx+1`` and is held until the
    flip at ``idx+2``, so the side compares the YES mid at ``idx+1`` and ``idx+2``;
    the limit is candle ``idx+1``'s low for that side (YES ask low, or NO ask low =
    ``1 - YES bid high``) so the fill is the best price available. close_time order.
    """
    candles = pd.read_csv(_DATA / "candlesticks.csv")
    plan: dict = {}
    for ticker, grp in candles.groupby("ticker"):
        g = grp.sort_values("close_time")
        mid = ((g["yes_ask_open"] + g["yes_bid_open"]) / 2.0).to_numpy()
        ask_low = g["yes_ask_low"].to_numpy()
        bid_high = g["yes_bid_high"].to_numpy()
        for idx in range(len(mid) - 2):
            delta = mid[idx + 2] - mid[idx + 1]
            if delta > 0:
                plan[(ticker, idx)] = ("yes", float(ask_low[idx + 1]))
            elif delta < 0:
                plan[(ticker, idx)] = ("no", float(1.0 - bid_high[idx + 1]))
    return plan


def test_oracle_and_perfect_are_upper_bounds(eng: Engine):
    oracle = _oracle()
    plan = _perfect_plan()
    correct, yes_limit, perfect = eng.run([_BuyCorrect(oracle), _BuyYesLimit(), _Perfect(plan)])

    # Oracle buys the winner every time -> essentially every market wins and net is
    # hugely positive. (A market priced at ~1.0 at entry can still net slightly
    # negative after the fee, so win_rate is ~1, not exactly 1.)
    assert correct["win_rate"] > 0.999
    assert correct["net_return"] > 0
    assert correct["fill_rate"] == pytest.approx(1.0)

    # Perfect rides every candle-to-candle move, capturing all the intra-market
    # volatility -> the most profitable strategy, beating even the buy-the-winner oracle.
    assert perfect["net_return"] > correct["net_return"]

    # Both touch the same markets, but Perfect flips many times per market while the
    # oracle fills once each -> Perfect racks up far more fills than markets. (The
    # oracle's fills can exceed markets_traded by one: the final market fills but is
    # never settled, so it counts as a fill but not a traded market.)
    assert perfect["markets_traded"] == correct["markets_traded"]
    assert correct["total_fills"] <= correct["markets_traded"] + 1
    assert perfect["total_fills"] > 2 * perfect["markets_traded"]

    # The YES-limit strategy exercises the resting maker path: orders are placed,
    # and (resting at a fixed limit) not all of them fill.
    assert set(yes_limit) == _KEYS
    assert eng.brokers[1].orders_submitted > 0
    assert yes_limit["fill_rate"] < 1.0
