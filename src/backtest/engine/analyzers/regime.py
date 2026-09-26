import numpy as np
import pandas as pd

from src.backtest.engine.analyzer import Analyzer, TradeReport

VOLUME_BASELINE_BARS = 96


def _curve_table(df: pd.DataFrame, col: str, n_bins: int = 12) -> pd.DataFrame:
    """Win-rate-vs-value curve: win rate + 95% Wilson CI per fine bin of ``col``.

    ``col`` is split into ``n_bins`` equal-count (quantile) bins; each row is one
    bin at its mean ``x`` value, so plotting ``win_rate`` against ``x`` traces how
    the fade's accuracy varies across the feature — and where it crosses the
    fee-adjusted breakeven (the natural place to draw a regime cutpoint).
    """
    from statsmodels.stats.proportion import proportion_confint

    cols = ["x", "n", "win_rate", "ci_lo", "ci_hi", "avg_pnl"]
    if len(df) < 2:
        return pd.DataFrame(columns=cols)
    try:
        bins = pd.qcut(df[col], q=min(n_bins, len(df)), duplicates="drop")
    except ValueError:                       # too few distinct values to bin
        return pd.DataFrame(columns=cols)
    rows = []
    for _, g in df.groupby(bins, observed=True):
        wins, n = int(g["won"].sum()), len(g)
        lo, hi = proportion_confint(wins, n, alpha=0.05, method="wilson")
        rows.append({"x": float(g[col].mean()), "n": n, "win_rate": wins / n,
                     "ci_lo": lo, "ci_hi": hi, "avg_pnl": float(g["pnl"].mean())})
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def logit_fit(df: pd.DataFrame, col: str) -> dict:
    """Logistic fit ``won ~ z + z^2`` on the **standardized** feature ``z``.

    The feature is z-scored first (mean 0, unit SD) so the design matrix is well
    conditioned — a raw trend (~1e-2) gives a ~1e-4 square term and huge, unstable
    coefficients (statsmodels fails to converge) otherwise. Coefficients are thus
    **per standard deviation**: the **linear** term is the directional tilt (up vs
    down), the **quadratic** the U-shape/strength effect (negative = the fade wins
    less as the feature grows extreme in *either* direction). Each carries a p-value
    (unchanged by scaling). NaNs if the feature is constant or the fit fails.
    """
    import statsmodels.api as sm

    nan = float("nan")
    fail = {"coef_linear": nan, "p_linear": nan, "coef_quad": nan,
            "p_quad": nan, "pseudo_r2": nan, "n": int(len(df))}
    x = df[col].to_numpy(dtype=float)
    sd = float(x.std())
    if sd == 0.0 or len(df) < 10:
        return fail
    z = (x - x.mean()) / sd   # standardize -> per-SD coefficients, well-conditioned
    X = sm.add_constant(np.column_stack([z, z ** 2]))
    try:
        res = sm.Logit(df["won"].to_numpy(dtype=float), X).fit(disp=0)
        return {"coef_linear": float(res.params[1]), "p_linear": float(res.pvalues[1]),
                "coef_quad": float(res.params[2]), "p_quad": float(res.pvalues[2]),
                "pseudo_r2": float(res.prsquared), "n": int(len(df))}
    except Exception:
        return fail


def dayofweek_table(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate + 95% Wilson CI per weekday, then a weekday/weekend split.

    Day of week is **cyclical**, so it gets a categorical table rather than the quantile
    binning the continuous features use: Mon=0..Sun=6 has no ordering to trace a curve
    along, and a ``won ~ d + d^2`` fit on it would be meaningless. Needs ``dayofweek``
    and ``won`` columns.
    """
    from statsmodels.stats.proportion import proportion_confint

    cols = ["bucket", "n", "win_rate", "ci_lo", "ci_hi"]
    if df.empty or "dayofweek" not in df.columns:
        return pd.DataFrame(columns=cols).set_index("bucket")

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    buckets = [(names[d], df["dayofweek"] == d) for d in range(7)]
    buckets += [("weekday", df["dayofweek"] < 5), ("weekend", df["dayofweek"] >= 5)]
    rows = []
    for label, mask in buckets:
        group = df[mask]
        if group.empty:
            continue
        wins, n = int(group["won"].sum()), len(group)
        lo, hi = proportion_confint(wins, n, alpha=0.05, method="wilson")
        rows.append({"bucket": label, "n": n, "win_rate": wins / n, "ci_lo": lo, "ci_hi": hi})
    if not rows:
        return pd.DataFrame(columns=cols).set_index("bucket")
    return pd.DataFrame(rows).set_index("bucket")


class RegimeAnalyzer(Analyzer):
    """Conditions a run's per-trade outcomes on BTC market regime.

    Trailing **trend** (window return), **|trend|** (strength), and **volatility**
    (std of log returns) are computed from an injected BTC OHLCV frame (columns
    ``close_time``, ``close``) for each ``windows`` entry (name -> number of bars),
    joined to the run's trades by time, and used to quantify *how* the fade's win
    rate varies with regime:

    * ``regime_curve`` — per feature, a **win-rate-vs-value curve** (fine quantile
      bins with 95% Wilson CIs), for reading where win rate crosses breakeven;
    * ``regime_logit`` — per feature, a **logistic fit** ``won ~ f + f^2`` (linear
      = direction, quadratic = curvature) with p-values;
    * ``regime_correlations`` — point-biserial ``corr(won, feature)``.

    (Replaces the earlier fixed tercile buckets, which forced arbitrary thirds; the
    curve locates the real cutpoints and the logit tests the asymmetry/U-shape.)

    Opt-in (reads external data; ``scipy``/``statsmodels`` are imported lazily so
    core engine imports stay light). ``won`` is size-independent, so the result
    does not depend on the run's Kelly sizing.
    """

    def __init__(self, btc: pd.DataFrame, windows: dict[str, int] | None = None,
                 n_bins: int = 12) -> None:
        self.btc = btc
        # Short horizons avoid presenting highly overlapping long windows as
        # independent evidence in a typical backtest.
        self.windows = windows or {"1h": 4, "3h": 12, "6h": 24, "24h": 96}
        self.n_bins = n_bins

    def _features(self) -> tuple[pd.DataFrame, list[str]]:
        missing = [c for c in ("close_time", "close", "volume") if c not in self.btc.columns]
        if missing:
            raise ValueError(
                f"price frame is missing {missing}; read it with "
                'usecols=["close_time", "close", "volume"] — volume_* features need it'
            )
        btc = self.btc[["close_time", "close", "volume"]].copy()
        btc["close_time"] = pd.to_datetime(btc["close_time"], utc=True)
        btc = btc.sort_values("close_time").reset_index(drop=True)
        btc["logret"] = np.log(btc["close"]).diff()
        baseline = btc["volume"].rolling(VOLUME_BASELINE_BARS).mean()
        feat_cols: list[str] = []
        for w, bars in self.windows.items():
            btc[f"trend_{w}"] = btc["close"].pct_change(bars)
            btc[f"volatility_{w}"] = btc["logret"].rolling(bars).std()
            # `.shift(bars)` ends the baseline just before the measured window, so the
            # window is never part of its own denominator. Without it a 6h window is 25%
            # of a 1d baseline, which damps the ratio toward 1 and costs a third of its
            # dynamic range (p90/p10 2.69x -> 3.53x with the shift).
            base = baseline.shift(bars)
            btc[f"volume_{w}"] = btc["volume"].rolling(bars).mean() / base.where(base > 0)
            feat_cols += [f"trend_{w}", f"volatility_{w}", f"volume_{w}"]
        # Calendar regime. Kept separate from the windowed features: day-of-week is
        # cyclical, so Mon=0..Sun=6 has no meaningful ordering to bin or fit a curve
        # through — it gets its own per-day table. `is_weekend` IS a proper binary, so
        # it can join the correlation table.
        btc["dayofweek"] = btc["close_time"].dt.dayofweek
        btc["is_weekend"] = (btc["dayofweek"] >= 5).astype(int)
        feat_cols += ["dayofweek", "is_weekend"]
        return btc, feat_cols

    def metrics(self, broker: TradeReport) -> dict:
        from scipy import stats

        trades = broker.get_trades()
        empty = {"regime_n": 0, "regime_features": pd.DataFrame(), "regime_curve": {},
                 "regime_logit": pd.DataFrame(), "regime_correlations": pd.DataFrame()}
        if not trades:
            return empty

        btc, feat_cols = self._features()
        trades_df = pd.DataFrame({
            "close_time": pd.to_datetime([t["close_time"] for t in trades], utc=True),
            "won": [int(t["won"]) for t in trades],
            "pnl": [t["pnl"] for t in trades],
        }).sort_values("close_time").reset_index(drop=True)

        # LOOKAHEAD-SAFE JOIN: match each trade to the last BTC bar STRICTLY BEFORE its
        # close_time (allow_exact_matches=False). A trade's close_time is the market's
        # close, whose bar is the *outcome* bar; a trailing window ending there would let
        # the settlement move leak into the regime feature (e.g. pct_change(24)'s endpoint
        # is the outcome bar's close — a losing fade inflates its own |trend|, manufacturing
        # a spurious "strong trend -> loses" effect). Excluding the exact match ends the
        # window at the signal/entry bar, so the feature is what was known at entry.
        regime = pd.merge_asof(
            trades_df, btc[["close_time", *feat_cols]], on="close_time",
            direction="backward", allow_exact_matches=False,
        ).dropna(subset=feat_cols).reset_index(drop=True)
        if regime.empty:
            return empty
        for w in self.windows:
            regime[f"abs_trend_{w}"] = regime[f"trend_{w}"].abs()

        # Signed trend first (up/down asymmetry drives the bucketing decision), then
        # |trend| (strength), volatility and relative volume. `dayofweek` is excluded —
        # it is cyclical, so binning it or fitting won ~ d + d^2 would be meaningless.
        analysis_feats = ([f"trend_{w}" for w in self.windows]
                          + [f"abs_trend_{w}" for w in self.windows]
                          + [f"volatility_{w}" for w in self.windows]
                          + [f"volume_{w}" for w in self.windows])
        curve = {f: _curve_table(regime, f, self.n_bins) for f in analysis_feats}
        logit = pd.DataFrame({f: logit_fit(regime, f) for f in analysis_feats}).T

        corr = pd.DataFrame(
            {f: dict(zip(("corr_with_win", "p_value"), stats.pointbiserialr(regime["won"], regime[f])))
             for f in [*analysis_feats, "is_weekend"]}
        ).T

        return {"regime_n": len(regime), "regime_features": regime, "regime_curve": curve,
                "regime_logit": logit, "regime_correlations": corr,
                "regime_dayofweek": dayofweek_table(regime)}
