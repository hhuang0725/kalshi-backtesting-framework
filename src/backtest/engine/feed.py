from typing import Any, Protocol, runtime_checkable

from src.backtest.engine.feeds.csv_feed import CsvFeed, load_columns
from src.backtest.engine.feeds.frame_feed import FrameFeed, FrameFeedView, Line, LineView
from src.backtest.engine.feeds.kalshi_market import MarketFeed

__all__ = [
    "CsvFeed",
    "Feed",
    "FeedView",
    "FrameFeed",
    "FrameFeedView",
    "Line",
    "LineView",
    "MarketFeed",
    "load_columns",
]


@runtime_checkable
class FeedView(Protocol):
    """The deliberately limited feed capability given to a strategy."""

    name: str

    def __len__(self) -> int: ...


@runtime_checkable
class Feed(Protocol):
    """Supplies time-ordered bars, revealing only those at or before the current time.

    The master feed sets the time each tick through :meth:`step`; other feeds catch
    up through :meth:`advance_to`. How a strategy reads values is defined by the
    concrete feed, so no column access appears here.

    Structural, so a database or a recorded live session can drive a run without
    inheriting anything.
    """

    name: str

    def __len__(self) -> int:
        """Number of bars currently visible to a strategy."""
        ...

    def reset(self) -> None:
        """Return to the start of history, before any bar is visible."""
        ...

    def size(self) -> int:
        """Total bars available. The engine runs this many ticks off the master."""
        ...

    def step(self) -> None:
        """Advance one bar. Called on the master feed only."""
        ...

    def advance_to(self, master_time: Any) -> None:
        """Make visible every bar at or before ``master_time``. Monotonic."""
        ...

    def current_time(self) -> Any:
        """Timestamp of the newest visible bar."""
        ...

    def strategy_view(self) -> FeedView:
        """Return the read capability exposed through :class:`Context`."""
        ...
