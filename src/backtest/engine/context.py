from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from src.backtest.engine.feed import Feed, FeedView

_NO_POSITIONS: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """The account's cash and positions at one instant.

    Built by the broker and read by the strategy through ``ctx.account``.
    ``positions`` is keyed by whatever the broker calls its instruments, ``"yes"``
    and ``"no"`` for Kalshi.

    Frozen and slotted because the engine builds these on the per-tick path.
    """

    cash: float
    positions: Mapping[str, float] = field(default=_NO_POSITIONS)

    def __post_init__(self) -> None:
        # `frozen` stops `positions` being rebound, not the dict behind it being
        # edited. Copy before wrapping, or the caller keeps a mutable handle on it.
        if not isinstance(self.positions, MappingProxyType):
            object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))

    def position(self, name: str) -> float:
        """Size held in ``name``, or ``0.0`` when there is none."""
        return self.positions.get(name, 0.0)


class Context:
    """Everything a strategy may read on the current tick.

    Each registered feed is reachable by its name (``ctx.candle``, ``ctx.price``,
    ``ctx.market``, …); columns are then indexed relative to now, e.g.
    ``ctx.candle.yes_ask_open[0]``. The account is an immutable
    :class:`AccountSnapshot` on ``ctx.account``, and ``ctx.time`` is the master
    feed's current timestamp.

    **A per-tick view, not a value to keep.** The engine reuses one context per
    strategy and updates it in place, so a stored context describes a later tick.

    ``balance`` / ``yes_contracts`` / ``no_contracts`` are Kalshi-named properties
    over the snapshot; venue-neutral code reads ``ctx.account``.
    """

    def __init__(
        self,
        feeds: dict[str, Feed],
        *,
        account: AccountSnapshot,
        time: Any = None,
    ) -> None:
        self._feeds = {name: feed.strategy_view() for name, feed in feeds.items()}
        self.time = time
        self.account = account

    @property
    def balance(self) -> float:
        return self.account.cash

    @property
    def yes_contracts(self) -> float:
        return self.account.position("yes")

    @property
    def no_contracts(self) -> float:
        return self.account.position("no")

    def __getattr__(self, name: str) -> FeedView:
        # Only reached when normal attribute lookup fails (i.e. not a feed name).
        if name == "_feeds":
            raise AttributeError(name)
        try:
            return self._feeds[name]
        except KeyError:
            raise AttributeError(f"no feed or attribute named '{name}'") from None
