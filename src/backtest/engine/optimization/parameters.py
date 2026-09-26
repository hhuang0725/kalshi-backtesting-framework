from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import optuna


class ParamSpec(Protocol):
    """A tunable parameter that knows how to sample itself from an Optuna trial.

    Each spec wraps Optuna's imperative ``trial.suggest_*`` API behind a uniform
    ``suggest(trial, name)`` so a strategy can declare its search space as data
    (``SEARCH_SPACE = {name: spec}``) and the optimizer can iterate any space
    without knowing the parameter names. Optuna still does the searching — these
    are thin declarations of *what* to search.
    """

    def suggest(self, trial: "optuna.Trial", name: str) -> Any: ...


@dataclass(frozen=True)
class Continuous:
    """A real-valued parameter in ``[low, high]`` (optionally log-scaled)."""

    low: float
    high: float
    log: bool = False

    def suggest(self, trial: "optuna.Trial", name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log)


@dataclass(frozen=True)
class Integer:
    """An integer parameter in ``[low, high]`` (inclusive)."""

    low: int
    high: int

    def suggest(self, trial: "optuna.Trial", name: str) -> int:
        return trial.suggest_int(name, self.low, self.high)


@dataclass(frozen=True)
class Categorical:
    """A discrete parameter chosen from a fixed, hashable set of ``choices``."""

    choices: tuple

    def suggest(self, trial: "optuna.Trial", name: str) -> Any:
        return trial.suggest_categorical(name, self.choices)


Space = dict[str, ParamSpec]


def sample(space: Space, trial: "optuna.Trial") -> dict[str, Any]:
    """Draw one value per parameter from ``trial`` — ready to splat into a strategy."""
    return {name: spec.suggest(trial, name) for name, spec in space.items()}


def grid(space: Space) -> dict[str, list]:
    """Build an exhaustive search grid for ``optuna.samplers.GridSampler``.

    Every spec must be :class:`Categorical` — a grid over unbounded continuous
    ranges is undefined.
    """
    out: dict[str, list] = {}
    for name, spec in space.items():
        if not isinstance(spec, Categorical):
            raise TypeError(
                f"grid() requires Categorical specs; {name!r} is {type(spec).__name__}"
            )
        out[name] = list(spec.choices)
    return out
