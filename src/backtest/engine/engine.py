import logging
from pathlib import Path

from src.backtest.engine.analyzer import Analyzer
from src.backtest.engine.analyzers import BaseAnalyzer
from src.backtest.engine.context import Context
from src.backtest.engine.execution import KalshiBroker
from src.backtest.engine.feed import CsvFeed, Feed, MarketFeed
from src.backtest.engine.strategy import Strategy

logger = logging.getLogger(__name__)


class Engine:
    """Runs the backtest: advances the clock and calls feeds, broker and strategy
    in order.

    The **first feed added is the master** and drives the clock; the rest chase its
    timestamp. Fills, fees and settlement live in the broker, so the engine never
    sees a market, a ticker or an outcome.

    Usage::

        engine = Engine()
        engine.add_data_feed(master_feed)   # first = master
        engine.add_data_feed(other_feed)
        engine.broker_factory = lambda: MyBroker(...)
        engine.add_analyzer(MyAnalyzer())
        metrics = engine.run([strategy])
    """

    def __init__(self) -> None:
        self._feeds: list[Feed] = []
        self.broker = None          # configured template; forked per strategy in run()
        self.broker_factory = None  # or a zero-arg callable, preferred over the template
        self.brokers: list = []     # per-strategy brokers from the last run (for inspection)
        self.analyzers: list[Analyzer] = []

    def add_data_feed(self, feed: Feed) -> None:
        """Register a feed. The first feed registered is the master clock."""
        if any(existing.name == feed.name for existing in self._feeds):
            raise ValueError(f"duplicate feed name {feed.name!r}; feed names are the strategy's keys")
        self._feeds.append(feed)

    def _new_broker(self):
        """One independent broker for one strategy.

        ``broker_factory`` is preferred; the ``broker`` template is the older form.
        """
        if self.broker_factory is not None:
            return self.broker_factory()
        return self.broker.fork()

    def add_analyzer(self, analyzer: Analyzer) -> None:
        """Attach a metrics producer; outputs merge into run() results in attach order."""
        self.analyzers.append(analyzer)

    def run(self, strategies: list[Strategy]) -> list[dict]:
        """Run every strategy over the full master history in one pass.

        Sequence: reset feeds and strategies, one broker per strategy, tick,
        finalize every broker, then the analyzers. Within a tick the broker runs
        *before* the strategy — that ordering is what makes an order eligible only
        on the tick after it is submitted.

        Returns one metrics dict per strategy, in order, with the analyzers' outputs
        merged in attach order (later keys win). The brokers stay on
        ``self.brokers`` for inspection.
        """
        if not self._feeds:
            raise ValueError("no feeds registered — call add_data_feed() first")
        if self.broker is None and self.broker_factory is None:
            raise ValueError("engine.broker or engine.broker_factory must be set before run()")

        for feed in self._feeds:
            feed.reset()
        for strategy in strategies:
            strategy.reset()
        self.brokers = [self._new_broker() for _ in strategies]

        master = self._feeds[0]
        aux = self._feeds[1:]
        feed_map = {feed.name: feed for feed in self._feeds}
        # One context per strategy, updated in place: it is a per-tick view by
        # contract, and rebuilding it every tick is on the hot path.
        contexts = [Context(feed_map, account=broker.account) for broker in self.brokers]
        work = list(zip(strategies, self.brokers, contexts))

        for _ in range(master.size()):
            master.step()
            master_time = master.current_time()
            for feed in aux:
                feed.advance_to(master_time)

            for strategy, broker, ctx in work:
                broker.update()
                ctx.time = master_time
                ctx.account = broker.account
                broker.receive_orders(strategy.next(ctx))

        # Analyzers read a finished broker, so every broker closes its run first.
        for broker in self.brokers:
            broker.finalize()

        results = []
        for broker in self.brokers:
            merged: dict = {}
            for analyzer in self.analyzers:
                merged.update(analyzer.metrics(broker))
            results.append(merged)

        for strategy, metrics in zip(strategies, results):
            logger.info(
                "[%s] markets=%d fills=%d win_rate=%.1f%% net=%.2f",
                type(strategy).__name__, metrics.get("markets_traded", 0),
                metrics.get("total_fills", 0), metrics.get("win_rate", 0.0) * 100,
                metrics.get("net_return", 0.0),
            )
        return results


_MASTER_DIR = Path("data/master")
_STARTING_BALANCE = 1000.0


def build_engine(
    data_dir: Path = _MASTER_DIR,
    starting_balance: float = _STARTING_BALANCE,
    candle_file: str = "kxbtc15m_candlesticks.csv",
    price_file: str = "btc_ohlcv.csv",
    market_file: str = "kxbtc15m_markets.csv",
) -> "Engine":
    """Wire the Kalshi 15m-market feeds + broker into a ready-to-run :class:`Engine`.

    The candle feed is the master clock; the price OHLCV feed is auxiliary; the
    market feed supplies outcomes for settlement. Filenames default to the
    KXBTC15M layout; pass e.g.
    ``candle_file="kxeth15m_candlesticks.csv", market_file="kxeth15m_markets.csv"``
    for another series, and ``price_file="eth_ohlcv.csv"`` for another Binance
    symbol. The price feed is registered as
    ``"price"`` whatever symbol backs it, which is how strategies reach it
    (``ctx.price.*``).
    """
    candle = CsvFeed("candle", data_dir / candle_file)
    price = CsvFeed("price", data_dir / price_file)
    market = MarketFeed("market", data_dir / market_file)

    engine = Engine()
    engine.add_data_feed(candle)        # first = master clock
    engine.add_data_feed(price)
    engine.add_data_feed(market)
    engine.broker = KalshiBroker(starting_balance, master_feed=candle, market_feed=market)
    engine.add_analyzer(BaseAnalyzer())
    return engine
