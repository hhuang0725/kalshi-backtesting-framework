import numpy as np

from src.backtest.engine.analyzer import Analyzer, TradeReport


def _trade_returns(trades: list[dict], starting_balance: float) -> np.ndarray:
    """Per-trade return fractions of a compounding book: ``pnl_i / equity_{i-1}``.

    Equity before trade ``i`` is ``starting_balance + cumsum(pnl)[:i]``, so the
    return of a trade is its P&L relative to the balance it was sized against.
    """
    if not trades:
        return np.array([], dtype=float)
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    equity_before = starting_balance + np.concatenate(([0.0], np.cumsum(pnl)[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(equity_before != 0.0, pnl / equity_before, 0.0)
    return returns


_NO_RECOVERY = {"ttr_p50": float("nan"), "ttr_p95": float("nan"),
                "ttr_deep_p50": float("nan"), "ttr_censored_pct": 0.0}


def _percentile_or_nan(values: np.ndarray, q: float) -> float:
    """``np.percentile`` over a possibly-empty selection.

    Every path being censored is a real outcome — the horizon is too short to see a
    recovery — rather than an error, so it reports NaN instead of raising.
    """
    return float(np.percentile(values, q)) if values.size else float("nan")


def _recovery_metrics(equity: np.ndarray, running_peak: np.ndarray,
                      drawdown: np.ndarray, path_max_dd: np.ndarray) -> dict:
    """Trades from each path's deepest trough back to the peak that preceded it.

    A path still under water when the horizon ends has no recovery time. It is
    counted as censored and left out of the percentiles rather than clamped to the
    horizon, which would understate the wait, or dropped silently, which would hide
    how often it happens. A path that never fell at all recovers in zero trades.
    """
    n_paths, horizon = equity.shape
    trough = drawdown.argmax(axis=1)
    target = running_peak[np.arange(n_paths), trough]
    # Strictly after the trough: without this a path sitting at its own peak would
    # "recover" at the trough itself.
    regained = (equity >= target[:, None]) & (np.arange(horizon)[None, :] > trough[:, None])
    recovered = regained.any(axis=1)

    never_fell = path_max_dd <= 0.0
    ttr = np.where(recovered, regained.argmax(axis=1) - trough, -1).astype(float)
    ttr[never_fell] = 0.0
    measured = recovered | never_fell
    deep = path_max_dd >= np.percentile(path_max_dd, 95)
    return {
        "ttr_p50": _percentile_or_nan(ttr[measured], 50),
        "ttr_p95": _percentile_or_nan(ttr[measured], 95),
        "ttr_deep_p50": _percentile_or_nan(ttr[deep & measured], 50),
        "ttr_censored_pct": float(np.mean(~measured) * 100.0),
    }


def bootstrap_risk(returns: np.ndarray, *, horizon: int | None = None, n_paths: int = 10_000,
                   seed: int = 42, ruin_threshold_frac: float = 0.1,
                   starting_balance: float = 1.0) -> dict:
    """Ruin, tail and recovery statistics from a series of per-trade **return fractions**.

    The whole of :class:`MonteCarloAnalyzer`, minus the broker. It is separate because a
    walk-forward has no broker: its trades come from several engine runs and are staked at a
    leverage solved afterwards, so the return series is built rather than recorded. Sharing the
    body is what keeps the two the same measurement — same resampler, same quantile, same
    definition of recovery — which is the only reason a venue figure and a walk-forward figure may
    be printed in the same table.

    ``returns[i]`` is what trade *i* did to the bankroll as a fraction of it, so a strategy whose
    stake varies per trade is expressed directly rather than through an average stake.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    start = float(starting_balance)
    if n == 0:
        return {"p_ruin": 0.0, "final_equity_p5": start, "max_drawdown_pct_p95": 0.0,
                **_NO_RECOVERY}

    rng = np.random.default_rng(seed)
    # (n_paths, horizon) resampled returns; growth factors (1 + r) compounded per path.
    sampled = returns[rng.integers(0, n, size=(n_paths, n if horizon is None else horizon))]
    equity = start * np.cumprod(1.0 + sampled, axis=1)

    running_min = np.minimum.accumulate(equity, axis=1)
    ruined = running_min[:, -1] <= ruin_threshold_frac * start
    # The running peak includes the opening balance: a path that falls from its first step is
    # measured from where it started, and recovery means regaining that start.
    running_peak = np.maximum(np.maximum.accumulate(equity, axis=1), start)
    # Fraction of peak, x100 -- never dollars. `running_peak > 0` because every growth factor is
    # positive under proportional sizing, so the division is safe.
    drawdown = (running_peak - equity) / running_peak
    path_max_dd = np.max(drawdown, axis=1) * 100.0

    return {
        "p_ruin": float(np.mean(ruined)),
        "final_equity_p5": float(np.percentile(equity[:, -1], 5)),
        "max_drawdown_pct_p95": float(np.percentile(path_max_dd, 95)),
        **_recovery_metrics(equity, running_peak, drawdown, path_max_dd),
    }


class MonteCarloAnalyzer(Analyzer):
    """Bootstrap ruin/tail estimates from a finished run's per-trade returns.

    Resamples the run's per-trade **return fractions** with replacement, replays
    them as a compounding equity path, and aggregates over ``n_paths`` draws.
    Seeded, so :meth:`metrics` is reproducible. Opt-in (expensive) — attach it in
    the notebook for a selected config, not during an :func:`optimize` study.

    ``ruin_threshold_frac`` is a **fraction of starting balance** (0-1): a path
    counts as ruined once its equity falls to or below
    ``ruin_threshold_frac * starting_balance``. This must be fraction-of-start,
    not an absolute-zero floor — position sizing here is proportional (a
    fraction of *current* equity per trade), so equity decays geometrically
    toward zero but can never exactly reach it; a ``0.0`` floor is therefore
    unreachable by construction and reports ``p_ruin == 0.0`` regardless of how
    aggressively you size, at every ``kelly_fraction``. Default ``0.1`` (down
    90% from start) is a practical "this bankroll is functionally dead" bar
    that a real compounding path can actually cross.

    ``max_drawdown_pct_p95`` is a **percentage of the running peak**, matching
    :class:`BaseAnalyzer`'s ``max_drawdown_pct``. It cannot be an absolute
    peak-to-trough amount: these paths compound, so a dollar figure measures how
    large the path grew rather than how far it fell, and is not comparable across
    leverage. It previously did exactly that and was unusable as a result.

    ``horizon`` is how many trades each simulated path contains, defaulting to the
    run's own length. It matters because max drawdown **grows with the number of
    trades**: a 2,000-trade venue backtest and a 30,000-bet walk-forward cannot be
    held to the same drawdown figure, and the gap between them says more about how
    long each ran than about how risky either is. Pinning the horizon — "the worst
    drawdown over a window of this many trades" — makes the number a property of the
    strategy rather than of the sample, so two studies of different lengths can be
    read against one another. A horizon shorter than the run also costs less memory
    and time than the default, since the simulated array is ``n_paths x horizon``.

    ``ttr_*`` are **times to recover, counted in trades**: from the trough of a
    path's deepest drawdown back to the peak preceding it. Depth alone is a poor
    measure of pain — a 50% drawdown made back within a month is not the business a
    two-year one is — but recovery is heavily right-skewed, so the median and the
    95th percentile are reported and never the mean. ``ttr_deep_p50`` conditions on
    the worst 5% of paths by depth: that is the recovery which pairs with
    ``max_drawdown_pct_p95``, and it is slower than the unconditional median because
    deep drawdowns are also long ones.

    ``ttr_censored_pct`` is the share of paths still under water when the horizon
    ends. Those paths have no recovery time and are excluded from the other
    ``ttr_*`` figures, which biases those figures fast — so the exclusion is
    reported rather than left implicit, and it is large at short horizons. When
    every path is censored the other ``ttr_*`` values are NaN.
    """

    def __init__(self, seed: int = 42, n_paths: int = 10_000, ruin_threshold_frac: float = 0.1,
                 horizon: int | None = None) -> None:
        if horizon is not None and horizon < 1:
            raise ValueError(f"horizon must be at least one trade, got {horizon}")
        self.seed = seed
        self.n_paths = n_paths
        self.ruin_threshold_frac = ruin_threshold_frac
        self.horizon = horizon

    def metrics(self, broker: TradeReport) -> dict:
        return bootstrap_risk(_trade_returns(broker.get_trades(), broker.starting_balance),
                              horizon=self.horizon, n_paths=self.n_paths, seed=self.seed,
                              ruin_threshold_frac=self.ruin_threshold_frac,
                              starting_balance=broker.starting_balance)
