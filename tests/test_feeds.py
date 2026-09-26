"""In-memory storage, CSV loading, validation, and immutable history."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine.feeds.csv_feed import CsvFeed, load_columns
from src.backtest.engine.feeds.frame_feed import FrameFeed
from src.backtest.engine.feeds.kalshi_market import MarketFeed
from tests.helpers import mt, ts


def _columns(n: int = 4) -> dict[str, np.ndarray]:
    return {
        "close_time": np.array([np.datetime64(f"2026-01-01T00:{i:02d}:00") for i in range(1, n + 1)]),
        "close": np.arange(float(n)),
    }


# --- storage without a file --------------------------------------------------------


@pytest.mark.unit
def test_frame_feed_needs_no_csv():
    """The separation, demonstrated: columns in, feed out, no file touched."""
    feed = FrameFeed("mem", _columns())
    feed.step()
    feed.step()
    assert len(feed) == 2
    assert feed.close[0] == 1.0
    assert feed.close[-1] == 0.0


@pytest.mark.unit
def test_frame_feed_rejects_a_missing_time_column():
    with pytest.raises(ValueError, match="no time column"):
        FrameFeed("mem", {"close": np.arange(3.0)})


@pytest.mark.unit
def test_there_is_no_way_to_read_a_whole_column():
    """A strategy holds the feed via its Context, so any whole-column accessor would
    hand it bars that have not happened - right beside the guard that blocks
    `close[1]`. past_values is the sanctioned bulk read and it stops at the cursor."""
    feed = FrameFeed("mem", _columns())
    feed.step()
    feed.step()
    assert not hasattr(feed, "column")
    with pytest.raises(IndexError, match="lookahead"):
        feed.close[1]
    assert list(feed.close.past_values) == [0.0]      # strictly before the cursor


# --- immutability ------------------------------------------------------------------


@pytest.mark.unit
def test_an_accidental_write_to_the_history_is_blocked():
    """Feeds are shared across every strategy in a batch, so a write would rewrite
    history for all of them."""
    feed = FrameFeed("mem", _columns())
    for _ in range(4):
        feed.step()
    with pytest.raises(ValueError, match="read-only"):
        feed.close.past_values[0] = 99.0


@pytest.mark.unit
def test_the_feed_owns_its_data_and_does_not_share_it_with_the_caller():
    """The property a `writeable=False` view would not give: a view shares memory
    both ways, so the caller's later write would land in the feed."""
    mine = np.arange(4.0)
    cols = _columns()
    cols["close"] = mine
    feed = FrameFeed("mem", cols)

    assert not np.shares_memory(feed.close.past_values, mine)
    mine[0] = 42.0                              # caller still owns their array
    feed.step()
    feed.step()
    assert feed.close.past_values[0] == 0.0     # and the feed did not move


@pytest.mark.unit
def test_what_a_strategy_is_given_cannot_be_unlocked():
    """numpy refuses to re-enable writes on a view whose base is read-only, and
    past_values is such a view. The owning array could be unlocked, which is why
    nothing hands it out."""
    feed = FrameFeed("mem", _columns())
    for _ in range(4):
        feed.step()
    with pytest.raises(ValueError, match="WRITEABLE"):
        feed.close.past_values.flags.writeable = True


# --- construction-time validation --------------------------------------------------


@pytest.mark.unit
def test_columns_of_different_lengths_are_rejected():
    """Otherwise this surfaces as an IndexError from a Line read many ticks later,
    pointing at the read rather than at the malformed feed."""
    with pytest.raises(ValueError, match="rows but"):
        FrameFeed("mem", {"close_time": _columns()["close_time"], "close": np.arange(2.0)})


@pytest.mark.unit
def test_unsorted_timestamps_are_rejected():
    """The silent one. advance_to stops at the first row past the cursor, so an
    out-of-order feed exposes fewer bars than it holds - or none - without raising."""
    cols = _columns()
    cols["close_time"] = cols["close_time"][[2, 0, 1, 3]]
    with pytest.raises(ValueError, match="not sorted oldest-first"):
        FrameFeed("mem", cols)


@pytest.mark.unit
def test_repeated_timestamps_are_allowed():
    """Non-decreasing, not strictly increasing: consecutive markets share a boundary
    instant and several candles can close together."""
    cols = _columns()
    cols["close_time"] = cols["close_time"][[0, 0, 1, 2]]
    assert FrameFeed("mem", cols).size() == 4


# --- CSV loading and validation ----------------------------------------------------


@pytest.mark.unit
def test_rows_are_sorted_oldest_first(tmp_path):
    """Load-bearing: data/master/*_ohlcv.csv are stored newest row first, and a feed
    that preserved that order would run every rolling window backwards through time."""
    path = tmp_path / "newest_first.csv"
    pd.DataFrame({"close_time": [ts(3), ts(2), ts(1)], "close": [3.0, 2.0, 1.0]}).to_csv(path, index=False)
    assert list(load_columns(path)["close"]) == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_missing_time_column_names_what_it_found(tmp_path):
    path = tmp_path / "no_time.csv"
    pd.DataFrame({"close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no time column"):
        load_columns(path)


@pytest.mark.unit
def test_an_unparseable_timestamp_is_rejected(tmp_path):
    """pandas raises on its own here; the wrapper adds which file it was."""
    path = tmp_path / "bad_time.csv"
    pd.DataFrame({"close_time": [ts(1), "not-a-date", ts(3)], "close": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unparseable close_time"):
        load_columns(path)


@pytest.mark.unit
def test_a_blank_timestamp_is_rejected(tmp_path):
    """Blanks become NaT instead of raising, so they need their own check — a row with
    no timestamp sorts to one end and shifts the clock without any error."""
    path = tmp_path / "blank_time.csv"
    pd.DataFrame({"close_time": [ts(1), None, ts(3)], "close": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing close_time"):
        load_columns(path)


@pytest.mark.unit
def test_data_feed_is_a_frame_feed_over_a_loaded_file(tmp_path):
    path = tmp_path / "feed.csv"
    pd.DataFrame({"close_time": [ts(1), ts(2)], "close": [1.0, 2.0]}).to_csv(path, index=False)
    feed = CsvFeed("f", path)
    assert isinstance(feed, FrameFeed)
    feed.advance_to(mt(2))
    assert feed.close[0] == 2.0


@pytest.mark.unit
def test_market_feed_masks_and_advances_strictly(tmp_path):
    """Both Kalshi-specific rules, now isolated in their own module."""
    path = tmp_path / "markets.csv"
    pd.DataFrame({
        "open_time": [ts(0), ts(2)], "close_time": [ts(2), ts(4)],
        "ticker": ["A", "B"], "result": ["yes", "no"],
    }).to_csv(path, index=False)
    feed = MarketFeed("market", path)
    feed.advance_to(mt(2))              # shared boundary instant: still market A
    assert feed.ticker[0] == "A"
    assert feed.result[0] is None       # masked while active
    feed.advance_to(mt(3))
    assert feed.ticker[0] == "B"
    assert feed.result[-1] == "yes"     # revealed once settled
