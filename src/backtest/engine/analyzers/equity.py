import numpy as np
import pandas as pd

from src.backtest.engine.analyzer import Analyzer, TradeReport
from src.backtest.engine.analyzers.baseanalyzer import _equity_curve


class EquityCurveAnalyzer(Analyzer):
    """Per-trade equity and drawdown *curves* for plotting a single run.

    Returns array-valued metrics, so it is opt-in (attach only when needed) and
    is not attached during an :func:`optimize` study — arrays would bloat the
    per-trial table.
    """

    def metrics(self, broker: TradeReport) -> dict:
        trades = broker.get_trades()
        equity = _equity_curve(trades, broker.starting_balance)
        # Includes the opening balance, as BaseAnalyzer does, so the two drawdowns agree.
        peak = (np.maximum(np.maximum.accumulate(equity), broker.starting_balance)
                if len(equity) else equity)
        drawdown = peak - equity
        return {
            "equity_curve": equity,
            "drawdown_curve": drawdown,
            "trade_times": [t["close_time"] for t in trades],
        }

    @staticmethod
    def plot(result: dict, btc: pd.DataFrame | None = None, log: bool = False, label: str = "BTC"):
        """Plot equity + drawdown on a time axis; overlay the price series' close if
        ``btc`` given.

        ``btc`` is the auxiliary price feed's frame — named for the original
        BTC use case, but any symbol's OHLCV works (e.g. ETH); ``label`` sets
        the chart title/legend/axis text to match (default "BTC", so existing
        callers need no changes). ``log=True`` puts the equity axis on a log
        scale (compounding reads as a straight line). Stateless — consumes the
        dict :meth:`metrics` returns. Returns the Figure.
        """
        import matplotlib.pyplot as plt

        equity = np.asarray(result["equity_curve"])
        drawdown = np.asarray(result["drawdown_curve"])
        times = result.get("trade_times")
        has_time = times is not None and len(times) > 0
        x = pd.to_datetime(pd.Series(times), utc=True) if has_time else np.arange(len(equity))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(x, equity, color="tab:blue", label="strategy equity")
        if log:
            ax1.set_yscale("log")
        ax1.set_ylabel("equity ($, log)" if log else "equity ($)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title("Equity curve" + (f" vs {label} price" if btc is not None else ""))
        ax1.grid(alpha=0.3)
        if btc is not None and has_time:
            b = btc[["close_time", "close"]].copy()
            b["close_time"] = pd.to_datetime(b["close_time"], utc=True)
            b = b.sort_values("close_time")
            b = b[(b["close_time"] >= x.min()) & (b["close_time"] <= x.max())]
            axb = ax1.twinx()
            axb.plot(b["close_time"], b["close"], color="tab:orange", alpha=0.6, label=f"{label} close")
            axb.set_ylabel(f"{label} close ($)", color="tab:orange")
            axb.tick_params(axis="y", labelcolor="tab:orange")
        ax2.fill_between(x, drawdown, color="tab:red", alpha=0.4)
        ax2.set_ylabel("drawdown ($)")
        ax2.set_xlabel("date" if has_time else "trade #")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        return fig
