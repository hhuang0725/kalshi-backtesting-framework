import pandas as pd
import pytest

from src.backtest.engine import Engine
from src.backtest.engine.analyzers import BaseAnalyzer
from src.backtest.engine.context import AccountSnapshot, Context
from src.backtest.engine.execution import KalshiBroker
from src.backtest.engine.feed import MarketFeed
from src.backtest.engine.strategies.high_probability import HighProbabilityStrategy
from tests.helpers import write_feed

pytestmark = pytest.mark.unit

_T0 = pd.Timestamp("2026-01-01T00:00:00+00:00")


def wts(i_open: int, minute: int = 0) -> str:
    """ISO timestamp ``minute`` minutes after 15m window boundary ``i_open``."""
    return (_T0 + pd.Timedelta(minutes=15 * i_open + minute)).isoformat()


def candle(ticker: str, i_open: int, minute: int,
           bid_high: float = 0.55, ask_low: float = 0.50) -> dict:
    """1m candle row for the given market window; defaults are quiet (no signal)."""
    return {
        "ticker": ticker,
        "open_time": wts(i_open, minute - 1),
        "close_time": wts(i_open, minute),
        "yes_bid_high": bid_high,
        "yes_ask_low": ask_low,
    }


def market_row(ticker: str, i_open: int, result: str = "yes") -> dict:
    return {
        "ticker": ticker, "open_time": wts(i_open),
        "close_time": wts(i_open + 1), "result": result,
    }


def make_feeds(tmp_path, candles: list[dict], markets: list[dict]):
    candle_feed = write_feed(tmp_path, "candle", candles)
    market_feed = write_feed(tmp_path, "market", markets, cls=MarketFeed)
    return candle_feed, market_feed


def tick(candle_feed, market_feed, strategy, yes: int = 0, no: int = 0):
    """Advance one master candle and call the strategy, mirroring Engine's loop."""
    candle_feed.advance()
    market_feed.advance(candle_feed.current_time())
    account = AccountSnapshot(cash=1000.0, positions={"yes": yes, "no": no})
    ctx = Context({"candle": candle_feed, "market": market_feed}, account=account)
    return strategy.next(ctx)


def run_all(tmp_path, candles, markets, strategy=None):
    """Tick through every candle; returns (orders per tick, strategy)."""
    strategy = strategy or HighProbabilityStrategy()
    candle_feed, market_feed = make_feeds(tmp_path, candles, markets)
    outs = [tick(candle_feed, market_feed, strategy) for _ in candles]
    return outs, strategy


def run_engine(tmp_path, candles, markets, strategy):
    candle_feed, market_feed = make_feeds(tmp_path, candles, markets)
    engine = Engine()
    engine.add_data_feed(candle_feed)       # master
    engine.add_data_feed(market_feed)
    engine.broker = KalshiBroker(1000.0, master_feed=candle_feed, market_feed=market_feed)
    engine.add_analyzer(BaseAnalyzer())
    (metrics,) = engine.run([strategy])
    return metrics


# --- signal detection ---

def test_yes_touch_emits_limit_order(tmp_path):
    candles = [candle("M1", 0, 1), candle("M1", 0, 2, bid_high=0.90)]
    outs, _ = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert outs[0] == []
    (order,) = outs[1]
    assert (order.side, order.quantity, order.order_type, order.limit_price) \
        == ("yes", 1, "limit", 0.90)


def test_no_touch_emits_no_limit_order(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.08, ask_low=0.10)]
    outs, _ = run_all(tmp_path, candles, [market_row("M1", 0)])
    (order,) = outs[0]
    assert (order.side, order.order_type, order.limit_price) == ("no", "limit", 0.90)


def test_quiet_market_no_orders(tmp_path):
    candles = [candle("M1", 0, m) for m in range(1, 16)]
    outs, strategy = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert all(out == [] for out in outs)
    assert strategy.signals == []


def test_threshold_touch_is_inclusive(tmp_path):
    # 0.899 stays below; exactly 0.90 fires (>= semantics)
    candles = [candle("M1", 0, 1, bid_high=0.899), candle("M1", 0, 2, bid_high=0.90)]
    outs, strategy = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert outs[0] == [] and len(outs[1]) == 1
    assert strategy.signals[0]["minute"] == 2


def test_yes_wins_when_both_sides_cross(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.95, ask_low=0.05)]
    outs, _ = run_all(tmp_path, candles, [market_row("M1", 0)])
    (order,) = outs[0]
    assert order.side == "yes"


def test_custom_threshold(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.85)]
    outs, _ = run_all(tmp_path, candles, [market_row("M1", 0)])    # default 0.90
    assert outs == [[]]
    outs, _ = run_all(tmp_path, candles, [market_row("M1", 0)],
                      strategy=HighProbabilityStrategy(threshold=0.85))
    (order,) = outs[0]
    assert order.limit_price == 0.85


def test_threshold_validation():
    for bad in (0.3, 0.5, 1.0, 1.2):
        with pytest.raises(ValueError):
            HighProbabilityStrategy(threshold=bad)


# --- one entry per market + signal log ---

def test_one_entry_per_market(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.92), candle("M1", 0, 2, bid_high=0.95)]
    outs, strategy = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert len(outs[0]) == 1 and outs[1] == []
    assert len(strategy.signals) == 1


def test_new_market_can_signal_again(tmp_path):
    candles = [
        candle("M1", 0, 1, bid_high=0.92),
        candle("M1", 0, 2),
        candle("M2", 1, 1, bid_high=0.91),
    ]
    markets = [market_row("M1", 0), market_row("M2", 1)]
    outs, strategy = run_all(tmp_path, candles, markets)
    assert len(outs[0]) == 1 and outs[1] == [] and len(outs[2]) == 1
    assert [s["ticker"] for s in strategy.signals] == ["M1", "M2"]


def test_signal_log_fields(tmp_path):
    candles = [candle("M1", 0, m) for m in range(1, 7)] + [candle("M1", 0, 7, bid_high=0.90)]
    _, strategy = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert strategy.signals == [{
        "ticker": "M1", "side": "yes", "minute": 7,
        "entry_price": 0.90, "filled": False,
    }]


def test_boundary_candle_belongs_to_closing_market(tmp_path):
    # The minute-15 candle closes exactly at M2's open; it must signal as M1, minute 15.
    candles = [candle("M1", 0, m) for m in range(1, 15)] \
        + [candle("M1", 0, 15, bid_high=0.90)]
    markets = [market_row("M1", 0), market_row("M2", 1)]
    outs, strategy = run_all(tmp_path, candles, markets)
    assert len(outs[-1]) == 1
    assert strategy.signals == [{
        "ticker": "M1", "side": "yes", "minute": 15,
        "entry_price": 0.90, "filled": False,
    }]


# --- guards ---

def test_candle_market_ticker_mismatch_no_order(tmp_path):
    candles = [candle("MX", 0, 1, bid_high=0.95)]
    outs, strategy = run_all(tmp_path, candles, [market_row("M1", 0)])
    assert outs == [[]] and strategy.signals == []


def test_no_active_market_yet_no_order(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.95)]
    outs, strategy = run_all(tmp_path, candles, [market_row("M1", 1)])
    assert outs == [[]] and strategy.signals == []


# --- fill observation ---

def test_fill_observed_from_yes_position(tmp_path):
    candles = [candle("M1", 0, 1, bid_high=0.92), candle("M1", 0, 2)]
    candle_feed, market_feed = make_feeds(tmp_path, candles, [market_row("M1", 0)])
    strategy = HighProbabilityStrategy()
    assert tick(candle_feed, market_feed, strategy) != []
    assert strategy.signals[0]["filled"] is False
    assert tick(candle_feed, market_feed, strategy, yes=1) == []
    assert strategy.signals[0]["filled"] is True


def test_fill_observation_requires_matching_side(tmp_path):
    candles = [
        candle("M1", 0, 1, bid_high=0.08, ask_low=0.10),    # NO signal
        candle("M1", 0, 2, bid_high=0.08, ask_low=0.12),
        candle("M1", 0, 3, bid_high=0.08, ask_low=0.12),
    ]
    candle_feed, market_feed = make_feeds(tmp_path, candles, [market_row("M1", 0)])
    strategy = HighProbabilityStrategy()
    assert tick(candle_feed, market_feed, strategy) != []
    tick(candle_feed, market_feed, strategy, yes=1)         # wrong side held
    assert strategy.signals[0]["filled"] is False
    tick(candle_feed, market_feed, strategy, no=1)
    assert strategy.signals[0]["filled"] is True


# --- end-to-end through Engine + KalshiBroker ---

def test_e2e_yes_fill_and_win(tmp_path):
    candles = [
        candle("M1", 0, 1),
        candle("M1", 0, 2, bid_high=0.90, ask_low=0.91),    # signal, no same-candle fill
        candle("M1", 0, 3, bid_high=0.89, ask_low=0.90),    # ask trades through -> fill
        candle("M1", 0, 4),
        candle("M2", 1, 1),                                 # next market -> M1 settles
    ]
    markets = [market_row("M1", 0, result="yes"), market_row("M2", 1)]
    strategy = HighProbabilityStrategy()
    metrics = run_engine(tmp_path, candles, markets, strategy)
    assert metrics["markets_traded"] == 1
    assert metrics["total_fills"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["net_return"] == pytest.approx(1.0 - 0.9 - 0.25 * 0.07 * 0.9 * 0.1)
    assert strategy.signals[0]["filled"] is True
    assert strategy.signals[0]["minute"] == 2


def test_e2e_yes_fill_and_loss(tmp_path):
    candles = [
        candle("M1", 0, 1, bid_high=0.90, ask_low=0.91),
        candle("M1", 0, 2, bid_high=0.89, ask_low=0.90),
        candle("M2", 1, 1),
    ]
    markets = [market_row("M1", 0, result="no"), market_row("M2", 1)]
    metrics = run_engine(tmp_path, candles, markets, HighProbabilityStrategy())
    assert metrics["markets_traded"] == 1
    assert metrics["win_rate"] == 0.0
    assert metrics["net_return"] == pytest.approx(-0.9 - 0.25 * 0.07 * 0.9 * 0.1)


def test_e2e_no_side_fill_and_win(tmp_path):
    candles = [
        candle("M1", 0, 1, bid_high=0.08, ask_low=0.10),    # NO signal
        candle("M1", 0, 2, bid_high=0.15, ask_low=0.20),    # no_ask = 1-0.15 <= 0.90 -> fill
        candle("M2", 1, 1),
    ]
    markets = [market_row("M1", 0, result="no"), market_row("M2", 1)]
    strategy = HighProbabilityStrategy()
    metrics = run_engine(tmp_path, candles, markets, strategy)
    assert metrics["markets_traded"] == 1
    assert metrics["win_rate"] == 1.0
    # NO fill at 0.90 is yes-denominated p=0.10; maker fee is symmetric in p
    assert metrics["net_return"] == pytest.approx(1.0 - 0.9 - 0.25 * 0.07 * 0.1 * 0.9)
    assert strategy.signals[0]["side"] == "no"
    assert strategy.signals[0]["filled"] is True


def test_e2e_unfilled_order_cancelled_at_close(tmp_path):
    candles = [
        candle("M1", 0, 1, bid_high=0.90, ask_low=0.93),    # signal
        candle("M1", 0, 2, bid_high=0.94, ask_low=0.95),    # ask never reaches 0.90
        candle("M2", 1, 1),     # quiet ask_low=0.50 would fill if not cancelled at settle
    ]
    markets = [market_row("M1", 0, result="yes"), market_row("M2", 1)]
    strategy = HighProbabilityStrategy()
    metrics = run_engine(tmp_path, candles, markets, strategy)
    assert metrics["markets_traded"] == 0
    assert metrics["total_fills"] == 0
    assert metrics["fill_rate"] == 0.0
    assert strategy.signals[0]["filled"] is False
