import numpy as np
import pandas as pd

from src.backtest.engine.feed import CsvFeed


def ts(minute: int) -> str:
    """ISO timestamp for a minute after 2026-01-01 00:00 UTC."""
    return f"2026-01-01T00:{minute:02d}:00+00:00"


def mt(minute: int) -> np.datetime64:
    """NumPy timestamp matching :func:`ts`."""
    return np.datetime64(f"2026-01-01T00:{minute:02d}:00")


def write_feed(tmp_path, name: str, rows: list[dict], cls=CsvFeed, **kwargs):
    """Write rows to a temporary CSV and construct a feed from it."""
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return cls(name, str(path), **kwargs)
