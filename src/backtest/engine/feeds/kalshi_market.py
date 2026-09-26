from pathlib import Path
from typing import Any

from src.backtest.engine.feeds.csv_feed import CsvFeed


class MarketFeed(CsvFeed):
    """Outcome feed keyed on ``open_time``, masking the current market's result.

    Advances on ``open_time < master_time`` (strict), so ``[0]`` is the current
    active market. Strict-on-open is required because consecutive markets share
    the boundary instant (``A.close_time == B.open_time``) and the candle closing
    there belongs to the *closing* market — strict-open keeps it at ``[0]``
    (masked) through its final candle and promotes the next market only on the
    following tick. The future-determined fields ``result``, ``expiration_value``,
    and ``volume_fp`` are masked at ``[0]`` and revealed once the market moves to ``[-1]``.
    """

    _MASK = ("result", "expiration_value", "volume_fp")

    def __init__(self, name: str, csv_path: str | Path) -> None:
        super().__init__(name, csv_path, time_col="open_time")

    def advance_to(self, master_time: Any) -> None:
        """Strict-open: a market stays current through the candle that closes it."""
        opens, n, size = self._times, self._n, self._size
        while n < size and opens[n] < master_time:
            n += 1
        self._n = n
