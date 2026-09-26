import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend for plot smoke tests

from src.backtest.engine.analyzers import (
    BaseAnalyzer,
    EquityCurveAnalyzer,
    MonteCarloAnalyzer,
    RegimeAnalyzer,
)


class _FakeBroker:
    """Just enough broker surface for an analyzer: trades + order counts + starting balance."""

    def __init__(
        self,
        trades: list[dict],
        orders_submitted: int,
        orders_filled: int,
        starting_balance: float = 1000.0,
    ) -> None:
        self.trades = trades
        self.orders_submitted = orders_submitted
        self.orders_filled = orders_filled
        self.starting_balance = starting_balance

    def get_trades(self) -> list[dict]:
        return list(self.trades)


def _trade(day: str, pnl: float, entry: float) -> dict:
    return {
        "close_time": pd.Timestamp(f"2026-01-{day}T00:15:00+00:00"),
        "pnl": pnl,
        "won": pnl > 0,
        "entry_price": entry,
        "contracts": 1,
    }


@pytest.mark.unit
def test_empty_trades_returns_zeros():
    m = BaseAnalyzer().metrics(_FakeBroker([], 0, 0))
    assert m["markets_traded"] == 0
    assert m["total_fills"] == 0
    assert m["win_rate"] == 0.0
    assert m["net_return"] == 0.0


@pytest.mark.unit
def test_basic_aggregates():
    trades = [_trade("01", 0.4, 0.6), _trade("02", -0.6, 0.4), _trade("03", 0.4, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, orders_submitted=4, orders_filled=3))
    assert m["markets_traded"] == 3
    assert m["total_fills"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["net_return"] == pytest.approx(0.2)
    assert m["fill_rate"] == pytest.approx(0.75)
    assert m["avg_entry_price"] == pytest.approx(0.5)


@pytest.mark.unit
def test_max_drawdown():
    # equity curve: +1, -2 (peak 1 -> trough -1 => drawdown 2), then +1
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3))
    assert m["max_drawdown"] == pytest.approx(2.0)


@pytest.mark.unit
def test_avg_pnl():
    trades = [_trade("01", 0.4, 0.6), _trade("02", -0.6, 0.4), _trade("03", 0.4, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3))
    assert m["avg_pnl"] == pytest.approx(0.2 / 3)


@pytest.mark.unit
def test_net_return_pct():
    trades = [_trade("01", 3.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3, starting_balance=100.0))
    assert m["net_return"] == pytest.approx(2.0)
    # net_return_pct is a true percentage: 2.0 / 100 * 100 = 2.0
    assert m["net_return_pct"] == pytest.approx(2.0)


@pytest.mark.unit
def test_net_return_pct_empty_is_zero():
    m = BaseAnalyzer().metrics(_FakeBroker([], 0, 0))
    assert m["net_return_pct"] == 0.0


@pytest.mark.unit
def test_avg_pnl_empty_is_zero():
    m = BaseAnalyzer().metrics(_FakeBroker([], 0, 0))
    assert m["avg_pnl"] == 0.0


@pytest.mark.unit
def test_flat_stake_ev():
    # flat-stake total EV = Σ(pnl / contracts): sizing-independent, frequency-aware.
    trades = [
        {"close_time": pd.Timestamp("2026-01-01T00:15:00+00:00"), "pnl": 4.0,
         "won": True, "entry_price": 0.5, "contracts": 8.0},   # 4/8 = 0.5
        {"close_time": pd.Timestamp("2026-01-02T00:15:00+00:00"), "pnl": -2.0,
         "won": False, "entry_price": 0.5, "contracts": 4.0},  # -2/4 = -0.5
        {"close_time": pd.Timestamp("2026-01-03T00:15:00+00:00"), "pnl": 3.0,
         "won": True, "entry_price": 0.5, "contracts": 2.0},   # 3/2 = 1.5
    ]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3))
    assert m["flat_stake_ev"] == pytest.approx(0.5 - 0.5 + 1.5)


@pytest.mark.unit
def test_flat_stake_ev_empty_is_zero():
    m = BaseAnalyzer().metrics(_FakeBroker([], 0, 0))
    assert m["flat_stake_ev"] == 0.0


@pytest.mark.unit
def test_min_equity_and_drawdown_pct():
    # start 100; equity path: 101, 99 (trough), 100. peak before trough = 101.
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3, starting_balance=100.0))
    assert m["min_equity"] == pytest.approx(99.0)
    # max relative drawdown as a percent = (101 - 99) / 101 * 100
    assert m["max_drawdown_pct"] == pytest.approx(2.0 / 101.0 * 100.0)


@pytest.mark.unit
def test_calmar():
    # net_return = 0.0 over the path, maxDD = 2.0 -> calmar 0.0
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 3, 3, starting_balance=100.0))
    assert m["calmar"] == pytest.approx(0.0 / 2.0)
    # positive net, known maxDD
    trades2 = [_trade("01", 3.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m2 = BaseAnalyzer().metrics(_FakeBroker(trades2, 3, 3, starting_balance=100.0))
    assert m2["net_return"] == pytest.approx(2.0)
    assert m2["calmar"] == pytest.approx(2.0 / 2.0)


@pytest.mark.unit
def test_calmar_zero_drawdown_is_zero():
    # monotonic-up curve has no drawdown; guarded to 0.0
    trades = [_trade("01", 1.0, 0.5), _trade("02", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 2, 2))
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["calmar"] == 0.0


@pytest.mark.unit
def test_equity_curve_analyzer():
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = EquityCurveAnalyzer().metrics(_FakeBroker(trades, 3, 3, starting_balance=100.0))
    assert list(np.asarray(m["equity_curve"])) == pytest.approx([101.0, 99.0, 100.0])
    # running peak - equity: peak=[101,101,101] -> [0, 2, 1]
    assert list(np.asarray(m["drawdown_curve"])) == pytest.approx([0.0, 2.0, 1.0])
    # per-trade timestamps come straight off the trade records
    assert list(m["trade_times"]) == [t["close_time"] for t in trades]


@pytest.mark.unit
def test_equity_curve_analyzer_empty():
    m = EquityCurveAnalyzer().metrics(_FakeBroker([], 0, 0, starting_balance=100.0))
    assert len(np.asarray(m["equity_curve"])) == 0
    assert len(np.asarray(m["drawdown_curve"])) == 0


@pytest.mark.unit
def test_montecarlo_seeded_determinism():
    trades = [_trade("01", 0.5, 0.5), _trade("02", -0.5, 0.5), _trade("03", 0.4, 0.5)]
    broker = _FakeBroker(trades, 3, 3, starting_balance=100.0)
    a = MonteCarloAnalyzer(seed=42, n_paths=500, ruin_threshold_frac=0.0)
    b = MonteCarloAnalyzer(seed=42, n_paths=500, ruin_threshold_frac=0.0)
    m1, m2 = a.metrics(broker), b.metrics(broker)
    assert m1["p_ruin"] == m2["p_ruin"]
    assert m1["final_equity_p5"] == pytest.approx(m2["final_equity_p5"])
    assert 0.0 <= m1["p_ruin"] <= 1.0


@pytest.mark.unit
def test_montecarlo_certain_ruin():
    # a single -100% return: any resampled path wipes the account out
    trades = [_trade("01", -100.0, 0.5)]  # r = -100/100 = -1.0
    broker = _FakeBroker(trades, 1, 1, starting_balance=100.0)
    m = MonteCarloAnalyzer(seed=1, n_paths=200, ruin_threshold_frac=0.0).metrics(broker)
    assert m["p_ruin"] == pytest.approx(1.0)


@pytest.mark.unit
def test_montecarlo_no_ruin_all_gains():
    trades = [_trade("01", 1.0, 0.5), _trade("02", 2.0, 0.5)]  # only positive returns
    broker = _FakeBroker(trades, 2, 2, starting_balance=100.0)
    m = MonteCarloAnalyzer(seed=7, n_paths=200, ruin_threshold_frac=0.0).metrics(broker)
    assert m["p_ruin"] == 0.0


@pytest.mark.unit
def test_montecarlo_empty_trades():
    broker = _FakeBroker([], 0, 0, starting_balance=100.0)
    m = MonteCarloAnalyzer(seed=1, n_paths=100, ruin_threshold_frac=0.0).metrics(broker)
    assert m["p_ruin"] == 0.0


@pytest.mark.unit
def test_montecarlo_drawdown_is_a_percentage_of_peak_not_dollars():
    """Regression: the bootstrap drawdown used to be an absolute peak-to-trough
    amount. On paths that compound, that measures how LARGE the path grew, not how
    far it fell — at kelly 0.85 it read ~1.1M on a 1,000 balance while the realized
    drawdown was 58%. A fraction of peak is bounded by 100% no matter how far the
    path runs; the dollar version is unbounded, so this test pins the unit."""
    # Strongly positive returns so paths compound hard, with one loss to create a dip.
    equity = 100.0
    trades = []
    for i, r in enumerate([0.6, 0.6, -0.3, 0.6, 0.6, 0.6], start=1):
        pnl = r * equity
        trades.append(_trade(f"{i:02d}", pnl, 0.5))
        equity += pnl
    broker = _FakeBroker(trades, len(trades), len(trades), starting_balance=100.0)

    m = MonteCarloAnalyzer(seed=3, n_paths=500, ruin_threshold_frac=0.1).metrics(broker)

    assert "max_drawdown_p95" not in m, "the dollar-valued key must be gone, not aliased"
    # On this fixture the old dollar version reported 314.6 against a starting balance
    # of 100 — a "drawdown" three times the whole account, on paths that never went
    # bankrupt. A fraction of peak cannot leave [0, 100]; here it is ~51%.
    assert 0.0 <= m["max_drawdown_pct_p95"] <= 100.0
    assert m["final_equity_p5"] > 100.0, "fixture must actually compound for this to bite"


@pytest.mark.unit
def test_montecarlo_zero_floor_is_structurally_unreachable_under_proportional_sizing():
    """Regression: position sizing here is proportional (a fraction of CURRENT
    equity per trade), so equity decays geometrically toward zero but can
    never exactly reach it — a literal ruin_threshold_frac=0.0 floor reports
    p_ruin=0.0 even for a path that has clearly lost almost everything. The
    default (0.1) must catch this; the old absolute-zero floor could not."""
    # Every trade loses 90% of equity AT THAT POINT -> after a few trades the
    # path has lost virtually everything (100 -> 10 -> 1 -> 0.1 -> ...), but
    # multiplicatively never hits exactly 0.
    equity = 100.0
    trades = []
    for i in range(1, 6):
        pnl = -0.9 * equity
        trades.append(_trade(f"{i:02d}", pnl, 0.5))
        equity += pnl
    broker = _FakeBroker(trades, len(trades), len(trades), starting_balance=100.0)

    zero_floor = MonteCarloAnalyzer(seed=1, n_paths=200, ruin_threshold_frac=0.0).metrics(broker)
    ten_pct_floor = MonteCarloAnalyzer(seed=1, n_paths=200, ruin_threshold_frac=0.1).metrics(broker)

    assert zero_floor["p_ruin"] == 0.0          # the old, blind-to-this-case behavior
    assert ten_pct_floor["p_ruin"] == pytest.approx(1.0)  # the fix: this path IS practically ruined


def _compounded(rates, start=100.0):
    """Trades whose P&L is a fixed fraction of equity *at that point*, so the run
    compounds the way a proportionally-sized book does."""
    equity, trades = start, []
    for i, r in enumerate(rates, start=1):
        pnl = r * equity
        trades.append(_trade(f"{i:02d}", pnl, 0.5))
        equity += pnl
    return _FakeBroker(trades, len(trades), len(trades), starting_balance=start)


# Slightly negative log-drift: wins and losses cancel arithmetically, so drawdown
# keeps deepening with the horizon instead of being outrun by growth.
_DRIFTLESS = [0.05, -0.05, 0.04, -0.04, 0.03, -0.03]


@pytest.mark.unit
def test_montecarlo_default_horizon_is_the_run_length():
    """The default must reproduce the pre-horizon numbers exactly. Anything else
    silently moves every stored notebook output and the golden pin for a parameter
    nobody set."""
    broker = _compounded(_DRIFTLESS)
    default = MonteCarloAnalyzer(seed=5, n_paths=300).metrics(broker)
    explicit = MonteCarloAnalyzer(seed=5, n_paths=300, horizon=len(_DRIFTLESS)).metrics(broker)
    assert set(default) == set(explicit)
    for key, got in default.items():
        want = explicit[key]
        # Exact, not approx -- and NaN-aware, since `NaN != NaN` would make two
        # identical result dicts compare unequal.
        assert (np.isnan(got) and np.isnan(want)) or got == want, key


@pytest.mark.unit
def test_montecarlo_horizon_must_be_at_least_one_trade():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least one trade"):
            MonteCarloAnalyzer(horizon=bad)


@pytest.mark.unit
def test_montecarlo_drawdown_grows_with_the_horizon():
    """The reason `horizon` exists: max drawdown is a function of how long you
    looked. Without pinning it, a 2,000-trade study and a 30,000-bet one cannot be
    held to the same number, and the difference measures run length rather than
    risk."""
    broker = _compounded(_DRIFTLESS)
    short = MonteCarloAnalyzer(seed=5, n_paths=400, horizon=50).metrics(broker)
    long = MonteCarloAnalyzer(seed=5, n_paths=400, horizon=1500).metrics(broker)
    assert long["max_drawdown_pct_p95"] > short["max_drawdown_pct_p95"] + 2.0


@pytest.mark.unit
def test_montecarlo_recovery_is_zero_when_the_path_never_falls():
    broker = _compounded([0.01, 0.02])          # only gains: nothing to recover from
    m = MonteCarloAnalyzer(seed=7, n_paths=200, horizon=20).metrics(broker)
    assert m["ttr_p50"] == 0.0
    assert m["ttr_censored_pct"] == 0.0


@pytest.mark.unit
def test_montecarlo_reports_censoring_rather_than_a_flattering_median():
    """Every path is still under water at the horizon. The honest answer is "no
    recovery observed", not a median computed from the handful that got closest —
    so the percentiles are NaN and the censoring is 100%."""
    broker = _compounded([-0.1] * 5)            # monotone down: the trough is the end
    m = MonteCarloAnalyzer(seed=1, n_paths=200, horizon=20).metrics(broker)
    assert m["ttr_censored_pct"] == pytest.approx(100.0)
    assert np.isnan(m["ttr_p50"]) and np.isnan(m["ttr_deep_p50"])


@pytest.mark.unit
def test_montecarlo_deep_drawdowns_recover_more_slowly():
    """`ttr_deep_p50` conditions on the worst 5% by depth — the recovery that pairs
    with `max_drawdown_pct_p95`. Depth and duration are not independent, so it must
    be the slower of the two."""
    broker = _compounded([0.06, -0.04, 0.05, -0.03, 0.06, -0.05])   # recovers on average
    m = MonteCarloAnalyzer(seed=5, n_paths=1000, horizon=400).metrics(broker)
    assert m["ttr_deep_p50"] >= m["ttr_p50"]
    assert not np.isnan(m["ttr_deep_p50"]), "fixture must let deep paths recover"


@pytest.mark.unit
def test_montecarlo_empty_run_reports_the_same_keys():
    """A caller that reads `ttr_p50` must not hit a KeyError purely because a config
    traded nothing."""
    empty = MonteCarloAnalyzer(seed=1, n_paths=100).metrics(
        _FakeBroker([], 0, 0, starting_balance=100.0))
    populated = MonteCarloAnalyzer(seed=1, n_paths=100).metrics(_compounded(_DRIFTLESS))
    assert set(empty) == set(populated)
    assert np.isnan(empty["ttr_p50"])


def _regime_fixture(seed=0):
    """Continuous BTC random walk + trades whose win prob falls with trailing |trend|.

    Guarantees a real negative relationship between |trend| and winning, and a
    smooth feature distribution (clean terciles for qcut).
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01T00:00:00+00:00")
    n = 1500
    times = [t0 + pd.Timedelta(minutes=15 * i) for i in range(n)]
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    # volume is required for the volume_* features; lognormal keeps it positive and
    # right-skewed like real 15m volume.
    btc = pd.DataFrame({"close_time": times, "close": close,
                        "volume": rng.lognormal(5.0, 0.6, n)})

    abs_trend = pd.Series(close).pct_change(4).abs()
    idx = np.arange(200, 1400, 4)
    med = abs_trend.iloc[idx].median()
    trades = []
    for i in idx:
        strong = abs_trend.iloc[i] > med
        won = bool(rng.random() < (0.30 if strong else 0.75))  # fade loses in strong trends
        trades.append({"close_time": times[i], "pnl": 0.5 if won else -0.5,
                       "won": won, "entry_price": 0.5, "contracts": 1})
    return btc, trades


@pytest.mark.unit
def test_regime_analyzer_curve_structure():
    btc, trades = _regime_fixture()
    m = RegimeAnalyzer(btc, windows={"1h": 4}, n_bins=8).metrics(
        _FakeBroker(trades, len(trades), len(trades)))

    # win-rate-vs-value curve per feature
    assert set(m["regime_curve"]) >= {"trend_1h", "abs_trend_1h", "volatility_1h", "volume_1h"}
    curve = m["regime_curve"]["trend_1h"]
    assert set(curve.columns) >= {"x", "n", "win_rate", "ci_lo", "ci_hi"}
    assert len(curve) >= 3
    assert curve["x"].is_monotonic_increasing            # bins ordered along the trend axis
    assert m["regime_n"] == len(trades)

    # the fade loses in strong trends → win negatively correlates with |trend|
    row = m["regime_correlations"].loc["abs_trend_1h"]
    assert row["corr_with_win"] < 0
    assert row["p_value"] < 0.05


@pytest.mark.unit
def test_regime_analyzer_logit_detects_u_shape():
    # Fixture: fade loses as |trend| grows (both signs), so win rate is an inverted-U in
    # signed trend → the quadratic term is negative and significant; linear ~ symmetric.
    btc, trades = _regime_fixture()
    m = RegimeAnalyzer(btc, windows={"1h": 4}).metrics(_FakeBroker(trades, len(trades), len(trades)))

    logit = m["regime_logit"]
    assert "trend_1h" in logit.index
    assert set(logit.columns) >= {"coef_linear", "p_linear", "coef_quad", "p_quad", "pseudo_r2"}
    assert logit.loc["trend_1h", "coef_quad"] < 0
    assert logit.loc["trend_1h", "p_quad"] < 0.05


@pytest.mark.unit
def test_regime_analyzer_empty_trades():
    btc, _ = _regime_fixture()
    m = RegimeAnalyzer(btc, windows={"1h": 4}).metrics(_FakeBroker([], 0, 0))
    assert m["regime_n"] == 0
    assert m["regime_curve"] == {}
    assert m["regime_logit"].empty


@pytest.mark.unit
def test_equity_curve_plot_smoke():
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5), _trade("03", 1.0, 0.5)]
    m = EquityCurveAnalyzer().metrics(_FakeBroker(trades, 3, 3, starting_balance=100.0))
    assert EquityCurveAnalyzer.plot(m) is not None
    btc, _ = _regime_fixture()
    assert EquityCurveAnalyzer.plot(m, btc=btc) is not None


@pytest.mark.unit
def test_equity_curve_plot_label_defaults_to_btc_but_is_overridable():
    """Regression: the title/legend/axis text used to hardcode "BTC" even when
    a different symbol's price frame was passed in (e.g. the ETH notebook)."""
    trades = [_trade("01", 1.0, 0.5), _trade("02", -2.0, 0.5)]
    m = EquityCurveAnalyzer().metrics(_FakeBroker(trades, 2, 2, starting_balance=100.0))
    price, _ = _regime_fixture()

    fig_default = EquityCurveAnalyzer.plot(m, btc=price)
    assert "BTC" in fig_default.axes[0].get_title()

    fig_eth = EquityCurveAnalyzer.plot(m, btc=price, label="ETH")
    assert "ETH" in fig_eth.axes[0].get_title()
    assert "BTC" not in fig_eth.axes[0].get_title()


@pytest.mark.unit
def test_a_loss_on_the_first_trade_counts_against_the_starting_balance():
    """Start at 100, lose 2, win 1: the account fell 2 below where it began. The P&L curve's first
    point is already -2, so a running peak seeded there reported no drawdown at all."""
    trades = [_trade("01", -2.0, 0.5), _trade("02", 1.0, 0.5)]
    m = BaseAnalyzer().metrics(_FakeBroker(trades, 2, 2, starting_balance=100.0))
    assert m["max_drawdown"] == pytest.approx(2.0)
    assert m["max_drawdown_pct"] == pytest.approx(2.0)
    curve = EquityCurveAnalyzer().metrics(_FakeBroker(trades, 2, 2, starting_balance=100.0))
    assert list(np.asarray(curve["drawdown_curve"])) == pytest.approx([2.0, 1.0])


@pytest.mark.unit
def test_bootstrap_depth_of_a_path_that_only_falls_is_measured_from_the_start():
    """Every resampled step loses 10%, so every path ends at 0.729 of its start: a 27.1% drawdown.
    A running peak seeded from the first post-trade balance reported 19%."""
    from src.backtest.engine.analyzers import bootstrap_risk
    m = bootstrap_risk(np.array([-0.1]), horizon=3, n_paths=50, seed=1)
    assert m["max_drawdown_pct_p95"] == pytest.approx(27.1)
