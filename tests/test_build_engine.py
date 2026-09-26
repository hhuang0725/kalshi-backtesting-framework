"""No-network unit tests for build_engine()'s filename parameterization.

Uses synthetic tmp-dir CSVs only (unlike test_engine.py's integration tests,
which need real data/master/*.csv), so this covers the ETH-vs-BTC filename
override path without depending on real market data.
"""
import pandas as pd
import pytest

from src.backtest.engine import build_engine
from src.backtest.engine.analyzers import BaseAnalyzer


def _write_master(tmp_path, candle_file: str, price_file: str, market_file: str) -> None:
    pd.DataFrame({
        "open_time": ["2026-01-01T00:00:00+00:00"],
        "close_time": ["2026-01-01T00:01:00+00:00"],
        "price_close": [0.5],
    }).to_csv(tmp_path / candle_file, index=False)

    pd.DataFrame({
        "open_time": ["2026-01-01T00:00:00+00:00"],
        "close_time": ["2026-01-01T00:15:00+00:00"],
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0],
    }).to_csv(tmp_path / price_file, index=False)

    pd.DataFrame({
        "open_time": ["2026-01-01T00:00:00+00:00"],
        "close_time": ["2026-01-01T00:15:00+00:00"],
        "result": ["yes"], "expiration_value": [100.5], "floor_strike": [100.0], "volume_fp": [1],
    }).to_csv(tmp_path / market_file, index=False)


@pytest.mark.unit
def test_build_engine_defaults_to_btc_filenames(tmp_path):
    _write_master(tmp_path, "kxbtc15m_candlesticks.csv", "btc_ohlcv.csv", "kxbtc15m_markets.csv")
    eng = build_engine(data_dir=tmp_path)
    assert [type(analyzer) for analyzer in eng.analyzers] == [BaseAnalyzer]
    eng.broker.master_feed.advance()
    assert eng.broker.master_feed.close_time[0] is not None  # candle feed loaded


@pytest.mark.unit
def test_build_engine_accepts_filename_overrides_for_eth(tmp_path):
    _write_master(tmp_path, "kxeth15m_candlesticks.csv", "eth_ohlcv.csv", "kxeth15m_markets.csv")
    eng = build_engine(
        data_dir=tmp_path,
        candle_file="kxeth15m_candlesticks.csv",
        price_file="eth_ohlcv.csv",
        market_file="kxeth15m_markets.csv",
    )
    price_feed = eng._feeds[1]
    # Registered as "price" whatever symbol backs it, so ctx.price is honest on ETH.
    assert price_feed.name == "price"
    price_feed.advance()
    assert price_feed.close[0] == 100.5


@pytest.mark.unit
def test_build_engine_missing_override_file_raises(tmp_path):
    _write_master(tmp_path, "kxbtc15m_candlesticks.csv", "btc_ohlcv.csv", "kxbtc15m_markets.csv")
    with pytest.raises(FileNotFoundError):
        build_engine(data_dir=tmp_path, price_file="eth_ohlcv.csv")  # not written in this tmp_path
