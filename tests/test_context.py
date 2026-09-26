import pytest

from src.backtest.engine.context import AccountSnapshot, Context
from tests.helpers import ts, write_feed


@pytest.mark.unit
def test_context_exposes_feeds_by_name(tmp_path):
    candle = write_feed(tmp_path, "candle", [{"close_time": ts(1), "x": 1}, {"close_time": ts(2), "x": 2}])
    price = write_feed(tmp_path, "price", [{"close_time": ts(1), "close": 100}])
    account = AccountSnapshot(cash=100.0, positions={"yes": 2.0, "no": 0.0})
    ctx = Context({"candle": candle, "price": price}, account=account)
    assert ctx.candle.name == candle.name
    assert ctx.price.name == price.name
    for control in ("step", "advance", "advance_to", "reset"):
        with pytest.raises(AttributeError):
            getattr(ctx.candle, control)
    assert ctx.balance == 100.0
    assert ctx.yes_contracts == 2
    with pytest.raises(AttributeError):
        _ = ctx.nonexistent_feed


@pytest.mark.unit
def test_context_feed_views_do_not_expose_storage_or_future_rows(tmp_path):
    candle = write_feed(
        tmp_path,
        "candle",
        [
            {"close_time": ts(1), "close": 1.0},
            {"close_time": ts(2), "close": 2.0},
            {"close_time": ts(3), "close": 3.0},
        ],
    )
    candle.step()
    candle.step()
    ctx = Context({"candle": candle}, account=AccountSnapshot(cash=100.0))

    for name in ("_columns", "_lines", "_source", "_feed", "_FrameFeedView__feed"):
        with pytest.raises(AttributeError):
            getattr(ctx.candle, name)
    for name in ("_values", "_line", "_LineView__line"):
        with pytest.raises(AttributeError):
            getattr(ctx.candle.close, name)

    history = ctx.candle.close.past_values
    assert history.tolist() == [1.0]
    assert history.base is None
    with pytest.raises(ValueError):
        history[0] = 99.0


@pytest.mark.unit
def test_context_requires_an_explicit_account_snapshot():
    with pytest.raises(TypeError):
        Context({})
