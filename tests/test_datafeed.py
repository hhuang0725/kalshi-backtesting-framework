import pytest

from src.backtest.engine.feed import MarketFeed
from tests.helpers import mt, ts, write_feed

# --- CsvFeed: line indexing + lookahead guard (master-style stepping) ---

@pytest.mark.unit
def test_line_indexing_current_and_previous(tmp_path):
    rows = [{"close_time": ts(1), "val": 10}, {"close_time": ts(2), "val": 20},
            {"close_time": ts(3), "val": 30}]
    f = write_feed(tmp_path, "f", rows)
    f.advance()
    f.advance()
    assert f.val[0] == 20       # current
    assert f.val[-1] == 10      # previous
    assert len(f) == 2


@pytest.mark.unit
def test_positive_index_raises_lookahead(tmp_path):
    f = write_feed(tmp_path, "f", [{"close_time": ts(1), "val": 1}, {"close_time": ts(2), "val": 2}])
    f.advance()
    with pytest.raises(IndexError):
        _ = f.val[1]


@pytest.mark.unit
def test_before_history_and_empty_raise(tmp_path):
    f = write_feed(tmp_path, "f", [{"close_time": ts(1), "val": 1}, {"close_time": ts(2), "val": 2}])
    f.advance()
    with pytest.raises(IndexError):
        _ = f.val[-1]
    f.reset()
    with pytest.raises(IndexError):
        _ = f.val[0]


@pytest.mark.unit
def test_advance_inclusive_of_boundary(tmp_path):
    rows = [{"close_time": ts(1), "val": 1}, {"close_time": ts(2), "val": 2}, {"close_time": ts(3), "val": 3}]
    f = write_feed(tmp_path, "f", rows)
    f.advance(mt(2))                # rows closing at <= 2 are visible
    assert len(f) == 2
    assert f.val[0] == 2           # the bar closing exactly at master_time is included


@pytest.mark.unit
def test_unknown_column_raises(tmp_path):
    f = write_feed(tmp_path, "f", [{"close_time": ts(1), "val": 1}])
    f.advance()
    with pytest.raises(AttributeError):
        _ = f.nonexistent


# --- MarketFeed: strict-open advance + masking ---

def _market_rows() -> list[dict]:
    return [
        {"ticker": "A", "open_time": ts(0), "close_time": ts(15),
         "result": "yes", "expiration_value": 1.0, "volume_fp": 3.0, "floor_strike": 2.0},
        {"ticker": "B", "open_time": ts(15), "close_time": ts(30),
         "result": "no", "expiration_value": 4.0, "volume_fp": 5.0, "floor_strike": 6.0},
    ]


@pytest.mark.unit
def test_marketfeed_strict_open_keeps_closing_market_at_boundary(tmp_path):
    f = write_feed(tmp_path, "market", _market_rows(), cls=MarketFeed)
    f.advance(mt(15))              # shared boundary instant: still market A
    assert f.ticker[0] == "A"
    assert len(f) == 1
    f.advance(mt(16))             # one minute later: market B is current
    assert f.ticker[0] == "B"
    assert f.ticker[-1] == "A"


@pytest.mark.unit
def test_marketfeed_masks_future_fields_on_current_only(tmp_path):
    f = write_feed(tmp_path, "market", _market_rows(), cls=MarketFeed)
    f.advance(mt(16))
    assert f.result[0] is None
    assert f.expiration_value[0] is None
    assert f.volume_fp[0] is None
    assert f.ticker[0] == "B"           # id is not masked
    assert f.result[-1] == "yes"        # previous market fully visible
    assert f.expiration_value[-1] == 1.0
    assert f.volume_fp[-1] == 3.0
