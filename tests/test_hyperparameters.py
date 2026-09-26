# external
import optuna
import pytest

# local
from src.backtest.engine.optimization.parameters import (
    Categorical,
    Continuous,
    Integer,
    grid,
    sample,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

pytestmark = pytest.mark.unit


def _trial() -> optuna.Trial:
    """A fresh ask-and-tell trial to drive ``.suggest`` against the real sampler."""
    return optuna.create_study().ask()


def test_continuous_suggests_within_range():
    spec = Continuous(1.5, 3.5)
    for _ in range(50):
        value = spec.suggest(_trial(), "x")
        assert 1.5 <= value <= 3.5
        assert isinstance(value, float)


def test_continuous_log_scale_stays_in_range():
    spec = Continuous(1e-4, 1e-1, log=True)
    for _ in range(50):
        value = spec.suggest(_trial(), "x")
        assert 1e-4 <= value <= 1e-1


def test_integer_suggests_inclusive_integers():
    spec = Integer(2, 5)
    seen = {spec.suggest(_trial(), "n") for _ in range(100)}
    assert seen <= {2, 3, 4, 5}
    assert all(isinstance(v, int) for v in seen)


def test_categorical_only_returns_choices():
    spec = Categorical((0.85, 0.90, 0.92, 0.95))
    seen = {spec.suggest(_trial(), "threshold") for _ in range(100)}
    assert seen <= {0.85, 0.90, 0.92, 0.95}


def test_specs_are_frozen():
    with pytest.raises(Exception):
        Continuous(0.0, 1.0).low = 2.0      # type: ignore[misc]


def test_sample_returns_all_keys():
    space = {"a": Continuous(0.0, 1.0), "b": Integer(1, 3), "c": Categorical(("x", "y"))}
    drawn = sample(space, _trial())
    assert set(drawn) == {"a", "b", "c"}
    assert 0.0 <= drawn["a"] <= 1.0
    assert drawn["b"] in (1, 2, 3)
    assert drawn["c"] in ("x", "y")


def test_grid_expands_categorical_space():
    space = {"threshold": Categorical((0.85, 0.90)), "q": Categorical((1, 2, 3))}
    assert grid(space) == {"threshold": [0.85, 0.90], "q": [1, 2, 3]}


def test_grid_rejects_non_categorical():
    with pytest.raises(TypeError, match="Categorical"):
        grid({"threshold": Continuous(0.8, 0.98)})
