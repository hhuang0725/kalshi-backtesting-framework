import math

import optuna
import pytest

from src.backtest.engine.optimization import OptimizationResult, optimize
from src.backtest.engine.optimization.parameters import Categorical, Continuous
from src.backtest.engine.strategy import Strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

_METRIC_KEYS = {
    "markets_traded", "total_fills", "win_rate", "net_return",
    "max_drawdown", "fill_rate", "avg_entry_price",
}


class _QuadStrategy(Strategy):
    """Toy strategy capturing a single ``threshold`` knob for the fake engine."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def next(self, ctx):
        return []


class _FakeEngine:
    """Scores a strategy from its params alone — no data, deterministic.

    ``net_return`` peaks at ``threshold == 0.9`` (an inverted parabola), so a
    working optimizer drives there. ``markets_traded`` can be forced below the
    constraint to exercise pruning.
    """

    def __init__(self, trades: int = 100, trades_fn=None) -> None:
        self._trades = trades
        self._trades_fn = trades_fn

    def run(self, strategies: list[_QuadStrategy]) -> list[dict]:
        out = []
        for s in strategies:
            trades = self._trades_fn(s.threshold) if self._trades_fn else self._trades
            out.append({
                "markets_traded": trades,
                "total_fills": trades,
                "win_rate": 0.5,
                "net_return": -((s.threshold - 0.9) ** 2),
                "max_drawdown": 0.0,
                "fill_rate": 1.0,
                "avg_entry_price": 0.5,
            })
        return out


pytestmark = pytest.mark.unit


def test_tpe_converges_near_optimum():
    space = {"threshold": Continuous(0.7, 1.1)}
    result = optimize(
        _QuadStrategy, space, n_trials=60, engine=_FakeEngine(), seed=42, batch_size=8,
    )
    assert isinstance(result, OptimizationResult)
    assert result.best_params["threshold"] == pytest.approx(0.9, abs=0.05)
    assert result.best_value == pytest.approx(0.0, abs=0.01)


def test_trials_dataframe_is_complete():
    space = {"threshold": Continuous(0.7, 1.1)}
    result = optimize(_QuadStrategy, space, n_trials=25, engine=_FakeEngine(), batch_size=8)
    assert len(result.trials) == 25
    assert "threshold" in result.trials.columns
    assert "objective" in result.trials.columns
    assert _METRIC_KEYS <= set(result.trials.columns)


def test_fixed_seed_is_deterministic():
    space = {"threshold": Continuous(0.7, 1.1)}
    a = optimize(_QuadStrategy, space, n_trials=30, engine=_FakeEngine(), seed=7)
    b = optimize(_QuadStrategy, space, n_trials=30, engine=_FakeEngine(), seed=7)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value


def test_all_under_min_trades_is_penalized():
    space = {"threshold": Continuous(0.7, 1.1)}
    result = optimize(
        _QuadStrategy, space, n_trials=10, engine=_FakeEngine(trades=5), min_trades=30,
    )
    assert math.isinf(result.best_value)
    assert result.best_value < 0       # -inf for a maximize run


def test_constraint_steers_away_from_under_powered_region():
    # net_return rises with threshold, but configs above 1.0 settle too few markets.
    space = {"threshold": Continuous(0.7, 1.2)}
    engine = _FakeEngine(trades_fn=lambda t: 100 if t <= 1.0 else 5)
    result = optimize(
        _QuadStrategy, space, n_trials=60, engine=engine, min_trades=30, seed=1,
    )
    assert result.best_metrics["markets_traded"] >= 30
    assert result.best_params["threshold"] <= 1.0 + 1e-9


def test_grid_sampler_is_exhaustive():
    space = {"threshold": Categorical((0.85, 0.90, 0.92, 0.95))}
    result = optimize(_QuadStrategy, space, sampler="grid", engine=_FakeEngine())
    assert len(result.trials) == 4
    assert set(result.trials["threshold"]) == {0.85, 0.90, 0.92, 0.95}
    assert result.best_params["threshold"] == 0.90      # closest to the 0.9 peak


def test_keep_strategies_returns_instances():
    space = {"threshold": Continuous(0.7, 1.1)}
    result = optimize(
        _QuadStrategy, space, n_trials=6, engine=_FakeEngine(), keep_strategies=True,
    )
    assert result.strategies is not None
    assert len(result.strategies) == 6
    assert all(isinstance(s, _QuadStrategy) for s in result.strategies)


def test_top_returns_best_first():
    space = {"threshold": Categorical((0.85, 0.90, 0.92, 0.95))}
    result = optimize(_QuadStrategy, space, sampler="grid", engine=_FakeEngine())
    top2 = result.top(2)
    assert len(top2) == 2
    assert top2.iloc[0]["objective"] >= top2.iloc[1]["objective"]
    assert top2.iloc[0]["threshold"] == 0.90


def test_tpe_requires_n_trials():
    with pytest.raises(ValueError, match="n_trials"):
        optimize(_QuadStrategy, {"threshold": Continuous(0.7, 1.1)}, engine=_FakeEngine())


def test_unknown_sampler_rejected():
    with pytest.raises(ValueError, match="sampler"):
        optimize(
            _QuadStrategy, {"threshold": Continuous(0.7, 1.1)},
            sampler="random", n_trials=5, engine=_FakeEngine(),
        )


def test_a_lambda_predicate_replaces_the_default_trade_floor():
    """The asymmetry this fixes: the objective was pluggable and validity was not."""
    space = {"threshold": Categorical((0.70, 0.90))}
    # 0.70 gets 5 fills, 0.90 gets 50; the predicate judges on fills, not markets.
    engine = _FakeEngine(trades_fn=lambda th: 5 if th < 0.8 else 50)
    result = optimize(_QuadStrategy, space, sampler="grid", engine=engine,
                      valid=lambda m: m["total_fills"] >= 10)
    assert result.best_params["threshold"] == 0.90
    assert math.isinf(result.trials["objective"].min())     # the 5-fill trial pruned


class _NoTradeCountEngine(_FakeEngine):
    """An analyzer set that does not report markets_traded."""

    def run(self, strategies):
        return [{k: v for k, v in m.items() if k != "markets_traded"}
                for m in super().run(strategies)]


def test_the_default_floor_names_the_remedy_when_its_metric_is_absent():
    space = {"threshold": Categorical((0.90,))}
    with pytest.raises(KeyError, match="pass valid="):
        optimize(_QuadStrategy, space, sampler="grid", engine=_NoTradeCountEngine())
