from typing import Protocol, runtime_checkable


@runtime_checkable
class TradeReport(Protocol):
    """The records a finished :class:`KalshiBroker` holds.

    What the analyzers in this module read. Stated as a type so an analyzer's
    dependency on one broker's report is visible rather than implied by an import.
    """

    starting_balance: float
    orders_submitted: int
    orders_filled: int

    def get_trades(self) -> list[dict]:
        """One record per settled position, in settlement order."""
        ...


class Analyzer:
    """Produces metrics from a finished broker.

    Analyzers attach to the engine (``engine.add_analyzer(...)``). After a run
    the engine calls :meth:`metrics` once per strategy-broker pair and merges
    the dicts in attach order. Analyzers are stateless with respect to the run —
    the broker is passed in rather than bound at attach time, so one instance
    serves every forked broker in a multi-strategy run.

    The engine calls them only after every broker is finalized. What a report
    contains is between an analyzer and the broker it was written for; the ones
    here read :class:`TradeReport`.
    """

    def metrics(self, broker: TradeReport) -> dict:
        raise NotImplementedError
