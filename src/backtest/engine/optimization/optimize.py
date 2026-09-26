import itertools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import optuna
import pandas as pd

from src.backtest.engine import build_engine
from src.backtest.engine.optimization.parameters import Space, grid, sample
from src.backtest.engine.strategy import Strategy

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MASTER_DIR = Path("data/master")
_DIRECTIONS = ("maximize", "minimize")
_SAMPLERS = ("tpe", "grid")


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of an :func:`optimize` run.

    ``trials`` holds one row per evaluated config (its sampled params, every
    backtest metric, and the ``objective`` score) in trial order — row ``i``
    aligns with ``strategies[i]`` when strategies were kept. Penalized trials
    (below ``min_trades``) carry an infinite ``objective`` so they sink in
    :meth:`top` and are never selected as best.
    """

    best_params: dict[str, Any]
    best_value: float
    best_metrics: dict[str, Any]
    trials: pd.DataFrame
    direction: str = "maximize"
    strategies: list[Strategy] | None = None

    def top(self, k: int = 5) -> pd.DataFrame:
        """The ``k`` best trials, best first (respecting the run's direction)."""
        return self.trials.sort_values(
            "objective", ascending=(self.direction == "minimize")
        ).head(k)


@dataclass
class _Trials:
    """Mutable accumulator for the per-trial record during a run."""

    params: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    strategies: list[Strategy] = field(default_factory=list)


def optimize(
    strategy_cls: type[Strategy],
    space: Space,
    *,
    n_trials: int | None = None,
    objective: Callable[[dict], float] | str = lambda m: m["net_return"],
    direction: str = "maximize",
    sampler: str = "tpe",
    min_trades: int = 30,
    valid: Callable[[dict], bool] | None = None,
    seed: int = 42,
    batch_size: int = 8,
    engine: Any = None,
    data_dir: Path = _MASTER_DIR,
    keep_strategies: bool = False,
) -> OptimizationResult:
    """Search ``space`` for the ``strategy_cls`` config that optimizes ``objective``.

    Each trial constructs ``strategy_cls(**params)`` and runs it over the full
    history; trials are batched into single ``engine.run([...])`` passes so the
    per-tick feed loop is amortized across the whole batch. ``sampler='tpe'``
    uses Optuna's TPE over continuous space (needs ``n_trials``); ``'grid'`` runs
    an exhaustive sweep over an all-``Categorical`` space (``n_trials`` ignored).
    Configs the sampler should not chase are pruned rather than scored. By
    default that means fewer than ``min_trades`` settled markets, which reads
    ``markets_traded`` from :class:`BaseAnalyzer`; pass ``valid`` to judge on
    anything else, e.g. ``valid=lambda m: m["total_fills"] >= 100``. A run with a
    different analyzer set needs it, since the default metric would be absent.

    Pass ``engine`` to inject a backtest runner (anything with
    ``run(list[Strategy]) -> list[dict]``); otherwise one is built from
    ``data_dir``. Set ``keep_strategies`` to retain the fitted instances (e.g. to
    read ``strategy.signals`` in a notebook).
    """
    if isinstance(objective, str):
        _key = objective
        objective = lambda m: m[_key]  # noqa: E731
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; use one of {_SAMPLERS}")
    if engine is None:
        engine = build_engine(data_dir)

    if valid is None:
        valid = _trade_floor(min_trades)

    acc = _Trials()
    runner = _BatchRunner(strategy_cls, engine, objective, valid, direction, keep_strategies, acc)

    if sampler == "grid":
        _run_grid(space, batch_size, runner)
    else:
        if n_trials is None:
            raise ValueError("n_trials is required for sampler='tpe'")
        _run_tpe(space, n_trials, batch_size, seed, direction, runner)

    return _build_result(acc, direction)


class _BatchRunner:
    """Evaluates a batch of param dicts: build strategies, run once, score, accumulate."""

    def __init__(self, strategy_cls, engine, objective, valid, direction, keep_strategies, acc):
        self._strategy_cls = strategy_cls
        self._engine = engine
        self._objective = objective
        self._valid = valid
        self._direction = direction
        self._keep = keep_strategies
        self._acc = acc

    def evaluate(self, params: list[dict]) -> list[tuple[float, bool]]:
        """Run ``params`` in one engine pass; return ``(score, valid)`` per param."""
        strategies = [self._strategy_cls(**p) for p in params]
        metrics = self._engine.run(strategies)
        out: list[tuple[float, bool]] = []
        for p, strat, m in zip(params, strategies, metrics):
            ok = self._valid(m)
            score = self._objective(m) if ok else _penalty(self._direction)
            self._acc.params.append(p)
            self._acc.metrics.append(m)
            self._acc.scores.append(score)
            if self._keep:
                self._acc.strategies.append(strat)
            out.append((score, ok))
        return out

    def best(self) -> float:
        return max(self._acc.scores) if self._direction == "maximize" else min(self._acc.scores)

    def count(self) -> int:
        return len(self._acc.scores)


def _run_grid(space: Space, batch_size: int, runner: "_BatchRunner") -> None:
    param_dicts = _grid_params(space)
    for batch in _batched(param_dicts, batch_size):
        runner.evaluate(batch)
        logger.info("optimize[grid] %d/%d trials, best=%.4f",
                    runner.count(), len(param_dicts), runner.best())


def _run_tpe(
    space: Space, n_trials: int, batch_size: int, seed: int, direction: str, runner: "_BatchRunner"
) -> None:
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    completed = 0
    while completed < n_trials:
        k = min(batch_size, n_trials - completed)
        trials = [study.ask() for _ in range(k)]
        params = [sample(space, t) for t in trials]
        results = runner.evaluate(params)
        for trial, (score, valid) in zip(trials, results):
            if valid:
                study.tell(trial, score)
            else:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        completed += k
        logger.info("optimize[tpe] %d/%d trials, best=%.4f", completed, n_trials, runner.best())


def _build_result(acc: _Trials, direction: str) -> OptimizationResult:
    arr = np.array(acc.scores, dtype=float)
    best_pos = int(arr.argmax() if direction == "maximize" else arr.argmin())
    trials_df = pd.DataFrame(
        [{**acc.params[i], **acc.metrics[i], "objective": acc.scores[i]} for i in range(len(acc.scores))]
    )
    return OptimizationResult(
        best_params=acc.params[best_pos],
        best_value=acc.scores[best_pos],
        best_metrics=acc.metrics[best_pos],
        trials=trials_df,
        direction=direction,
        strategies=acc.strategies if acc.strategies else None,
    )


def _grid_params(space: Space) -> list[dict]:
    """Every combination of an all-``Categorical`` space, in product order."""
    search_grid = grid(space)
    names = list(search_grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*search_grid.values())]


def _batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _trade_floor(min_trades: int) -> Callable[[dict], bool]:
    """The default validity rule: enough settled markets to be worth believing.

    Raises rather than treating a missing metric as zero, which would prune every
    trial and look like a space with no good configs.
    """
    def enough(metrics: dict) -> bool:
        try:
            return metrics["markets_traded"] >= min_trades
        except KeyError:
            raise KeyError(
                "optimize's default trade floor reads BaseAnalyzer's 'markets_traded', "
                "which this analyzer set does not produce; pass valid=... instead"
            ) from None

    return enough


def _penalty(direction: str) -> float:
    return -math.inf if direction == "maximize" else math.inf
