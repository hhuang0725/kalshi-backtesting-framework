from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine.feeds.frame_feed import FrameFeed

_DATETIME_COLS = ("open_time", "close_time")


def load_columns(csv_path: str | Path, time_col: str = "close_time") -> dict[str, np.ndarray]:
    """Read a CSV into ``{column: numpy array}``, sorted oldest-first by ``time_col``.

    The framework's only file boundary, so it is also where a malformed file is
    caught. The sort is load-bearing rather than tidiness: ``data/master/*_ohlcv.csv``
    are stored **newest row first**, and reading one unsorted runs every window
    backwards through time.
    """
    frame = pd.read_csv(csv_path)
    if time_col not in frame.columns:
        raise ValueError(f"{csv_path} has no time column {time_col!r}; found {list(frame.columns)}")
    for col in _DATETIME_COLS:
        if col in frame.columns:
            # tz-naive UTC, so columns compare against the engine's master_time.
            try:
                frame[col] = pd.to_datetime(frame[col], utc=True).dt.tz_localize(None)
            except ValueError as exc:
                raise ValueError(f"{csv_path} has an unparseable {col} value: {exc}") from exc
    # Blanks become NaT rather than raising above, and would sort to one end.
    if frame[time_col].isna().any():
        n = int(frame[time_col].isna().sum())
        raise ValueError(f"{csv_path} has {n} missing {time_col} value(s); "
                         "a row with no timestamp cannot be ordered")
    frame = frame.sort_values(time_col).reset_index(drop=True)
    return {col: frame[col].to_numpy() for col in frame.columns}


class CsvFeed(FrameFeed):
    """A :class:`FrameFeed` filled from a CSV file.

    What ``build_engine`` constructs. The only feed that reads a file.
    """

    def __init__(self, name: str, csv_path: str | Path, time_col: str = "close_time") -> None:
        super().__init__(name, load_columns(csv_path, time_col), time_col)
