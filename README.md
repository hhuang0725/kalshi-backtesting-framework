# Kalshi Backtesting Framework

A Python backtesting platform inspired by Backtrader, built specifically for
my own prediction-market research.

This repository contains the complete, standalone engine: its coordinator,
interfaces, feeds, simulated broker, execution policies, analyzers, optimizer,
example strategy, and tests. It does not include private research, datasets,
notebooks, or live-trading code.

The core `Engine` is strategy-, feed-, and broker-independent.
`build_engine()` is the included Kalshi-specific composition root.

## Lifecycle

For each run, the engine:

1. Resets every feed and strategy and creates an independent broker for each
   strategy.
2. Advances the first registered feed as the master clock, then advances every
   other feed to the same time.
3. Updates each broker, allowing previously submitted orders to fill and
   completed markets to settle.
4. Gives each strategy a `Context` containing the current time, an immutable
   account snapshot, and read-only views of every feed.
5. Sends the strategy's returned orders to its broker.
6. Finalizes each broker and runs the attached analyzers against it.

Broker updates happen before strategy evaluation, so an order cannot fill on
the same tick that created it. Analyzer outputs are merged into one metrics
dictionary per strategy, and `Engine.run()` returns those dictionaries in
strategy order.

## Architecture

```text
src/backtest/engine/
|-- engine.py          run coordinator and build_engine()
|-- feed.py            feed protocol and read-only strategy views
|-- broker.py          broker protocol
|-- strategy.py        strategy lifecycle and decision interface
|-- analyzer.py        finished-broker analyzer protocol
|-- context.py         per-tick context and immutable account snapshots
|-- orders.py          order and fill values
|-- feeds/             in-memory, CSV, and Kalshi market feeds
|-- execution/         account, fill, fee, lot, and settlement policies
|-- analyzers/         finished-broker reports
|-- optimization/      parameter declarations and batched optimization
`-- strategies/        example strategies
```

Dependencies point toward the framework contracts. The coordinator does not
import a concrete strategy. Strategies receive read-only feed views, while the
engine alone controls shared feed traversal.

## Extension points

- Implement `Feed` structurally to supply another data source. Its
  `strategy_view()` returns a separate object containing only the bounded read
  operations strategies may use, without cursor controls or backing storage.
- Implement `Broker` and expose an `AccountSnapshot` to model another venue or
  asset class.
- Subclass `Strategy`, clear per-run state in `reset()`, and return `Order`
  values from `next()`.
- Implement `Analyzer.metrics(finished_broker)` to produce an end-of-run
  report.
- Replace the included simulated broker's fill, fee, lot-size, settlement, or
  account policies without changing `Engine`.

See the protocols and tests for executable examples, including a fully generic
non-Kalshi feed, broker, strategy, and analyzer.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m ruff check src tests
python -m pytest
```

All included tests use synthetic or temporary data. No market datasets are
distributed with this repository.
