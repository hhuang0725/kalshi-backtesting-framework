from typing import Any

import numpy as np


class Line:
    """One column of a feed, indexed relative to the current bar.

    ``line[0]`` is the current bar, ``line[-1]`` the previous. A positive index
    raises instead of returning a bar that has not happened. A masked line returns
    ``None`` at ``[0]``, which hides an in-progress market's outcome until it settles.
    """

    def __init__(self, feed: "FrameFeed", values: Any, masked: bool = False) -> None:
        self._feed = feed
        self._values = values
        self._masked = masked

    def __getitem__(self, ago: int) -> Any:
        if ago > 0:
            raise IndexError("lookahead: positive indices are not allowed")
        n = self._feed._n
        if n == 0:
            raise IndexError(f"feed '{self._feed.name}' has no visible bars at this tick")
        idx = n - 1 + ago
        if idx < 0:
            raise IndexError("index reaches before the start of visible history")
        if self._masked and ago == 0:
            return None
        return self._values[idx]

    @property
    def past_values(self) -> Any:
        """Everything strictly before ``[0]``, oldest first, as a read-only view.

        The bulk read for numeric loops and numba kernels that cannot afford
        per-element ``__getitem__``; ``past_values[-k]`` equals ``line[-k]``.
        Stopping short of ``[0]`` is also what keeps a masked value from leaking.
        """
        n = self._feed._n
        return self._values[: max(n - 1, 0)]


class LineView:
    """Strategy-facing relative reads without access to a column's future rows."""

    __slots__ = ("__line",)

    def __init__(self, line: Line) -> None:
        object.__setattr__(self, "_LineView__line", line)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"line view has no attribute {name!r}")
        return object.__getattribute__(self, name)

    def __getitem__(self, ago: int) -> Any:
        line = object.__getattribute__(self, "_LineView__line")
        return line[ago]

    @property
    def past_values(self) -> Any:
        """Visible history as an isolated, read-only array."""
        line = object.__getattribute__(self, "_LineView__line")
        values = line.past_values.copy()
        values.flags.writeable = False
        return values


class FrameFeedView:
    """Expose a frame feed's named lines without its storage or cursor controls."""

    __slots__ = ("__feed", "__lines")

    def __init__(self, feed: "FrameFeed") -> None:
        object.__setattr__(self, "_FrameFeedView__feed", feed)
        lines = {name: LineView(line) for name, line in feed._lines.items()}
        object.__setattr__(self, "_FrameFeedView__lines", lines)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"feed view has no attribute {name!r}")
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            lines = object.__getattribute__(self, "_FrameFeedView__lines")
            try:
                return lines[name]
            except KeyError:
                raise AttributeError(f"feed view has no column or attribute {name!r}") from None

    @property
    def name(self) -> str:
        feed = object.__getattribute__(self, "_FrameFeedView__feed")
        return feed.name

    def __len__(self) -> int:
        feed = object.__getattribute__(self, "_FrameFeedView__feed")
        return len(feed)


class FrameFeed:
    """A feed over columns held in memory.

    :class:`CsvFeed` fills one from a file; :class:`MarketFeed` adds Kalshi's rules
    on top. ``feed.close`` returns the :class:`Line` for that column. :meth:`step`
    advances the master feed one bar; :meth:`advance_to` moves other feeds to the
    master's time.

    Columns are copied and frozen on construction, which makes
    ``Line.past_values`` a read-only view that cannot be unlocked.
    """

    _MASK: tuple[str, ...] = ()     # columns hidden at [0]; set by subclasses

    def __init__(self, name: str, columns: dict[str, np.ndarray], time_col: str = "close_time") -> None:
        _validate(name, columns, time_col)
        self.name = name
        self.time_col = time_col
        self._columns = {col: _frozen(arr) for col, arr in columns.items()}
        self._lines = {col: Line(self, arr, col in self._MASK) for col, arr in self._columns.items()}
        # Hoisted: advance_to runs once per feed per tick, where a dict lookup costs.
        self._times = self._columns[time_col]
        self._size = len(self._times)
        self._n = 0
        self._strategy_view = FrameFeedView(self)

    def strategy_view(self) -> FrameFeedView:
        """Return the cached column-only capability used by strategies."""
        return self._strategy_view

    def reset(self) -> None:
        self._n = 0

    def size(self) -> int:
        """Total rows. The engine runs this many ticks off the master feed."""
        return self._size

    def step(self) -> None:
        """Advance one bar, as the master clock."""
        self._n += 1

    def advance_to(self, master_time: Any) -> None:
        """Extend the window to every bar closing at or before ``master_time``."""
        times, n, size = self._times, self._n, self._size
        while n < size and times[n] <= master_time:
            n += 1
        self._n = n

    def advance(self, master_time: Any | None = None) -> None:
        """Dispatch to :meth:`step` or :meth:`advance_to`, for callers not yet migrated.

        The optional argument used to decide which of the two this meant, which is
        why they are now named separately.
        """
        if master_time is None:
            self.step()
        else:
            self.advance_to(master_time)

    def current_time(self) -> Any:
        return self._times[self._n - 1]

    def __len__(self) -> int:
        return self._n

    def __getattr__(self, name: str) -> Line:
        # Only reached when normal lookup fails (i.e. a column-line name).
        lines = self.__dict__.get("_lines")
        if lines is not None and name in lines:
            return lines[name]
        raise AttributeError(f"feed has no column or attribute '{name}'")


def _validate(name: str, columns: dict[str, np.ndarray], time_col: str) -> None:
    """Reject columns that :meth:`FrameFeed.advance_to` and the lines cannot serve.

    Runs once at construction so a malformed feed fails here rather than as a
    confusing read much later. Unsorted times would not raise at all on their own:
    ``advance_to`` stops at the first row past the cursor, so an out-of-order feed
    silently shows fewer bars than it holds.
    """
    if time_col not in columns:
        raise ValueError(f"feed '{name}' has no time column {time_col!r}")
    times = columns[time_col]
    for col, arr in columns.items():
        if len(arr) != len(times):
            raise ValueError(
                f"feed '{name}' column {col!r} has {len(arr)} rows but {time_col!r} has "
                f"{len(times)}; every column must cover the same bars"
            )
    # Non-decreasing, not strictly increasing: consecutive markets share a boundary
    # instant and several candles can close together.
    if len(times) > 1 and not (np.diff(times) >= 0).all():
        raise ValueError(
            f"feed '{name}' column {time_col!r} is not sorted oldest-first; "
            "advance_to would expose the wrong bars. Sort before constructing the feed"
        )


def _frozen(arr: np.ndarray) -> np.ndarray:
    """A read-only copy used to provide context to strategies.

    Copied rather than viewed because a read-only view of a writable array can be
    unlocked again, which would leave the guard on ``past_values`` decorative.
    """
    copy = arr.copy()
    copy.flags.writeable = False
    return copy
