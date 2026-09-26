import numpy as np

from src.backtest.engine.analyzer import Analyzer, TradeReport


def _equity_curve(trades: list[dict], starting_balance: float) -> np.ndarray:
    """Account equity after each trade: ``starting_balance + cumsum(pnl)``."""
    return starting_balance + np.cumsum([t["pnl"] for t in trades])


def _max_drawdown(trades: list[dict]) -> float:
    """Largest peak-to-trough drop of the cumulative P&L curve (positive).

    The running peak starts at zero P&L, the opening balance, so a loss on the first trade counts.
    """
    equity = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)
    return float(np.max(peak - equity)) if len(equity) else 0.0


class BaseAnalyzer(Analyzer):
    """Core trade + risk metrics: win rate, net return, drawdown, Calmar, fills."""

    def metrics(self, broker: TradeReport) -> dict:
        trades = broker.get_trades()
        n = len(trades)
        if n == 0:
            return {
                "markets_traded": 0,
                "total_fills": broker.orders_filled,
                "win_rate": 0.0,
                "net_return": 0.0,
                "net_return_pct": 0.0,
                "avg_pnl": 0.0,
                "flat_stake_ev": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "calmar": 0.0,
                "min_equity": broker.starting_balance,
                "fill_rate": 0.0,
                "avg_entry_price": 0.0,
            }

        net_return = sum(t["pnl"] for t in trades)
        max_dd = _max_drawdown(trades)
        equity = _equity_curve(trades, broker.starting_balance)
        # The running peak includes the opening balance, so a loss on the first trade is a drawdown,
        # and peak >= starting_balance > 0 keeps the division well-defined.
        peak = np.maximum(np.maximum.accumulate(equity), broker.starting_balance)
        # `*_pct` metrics are stored as true percentages (x100); win_rate stays a fraction.
        max_dd_pct = float(np.max((peak - equity) / peak)) * 100.0
        return {
            "markets_traded": n,
            "total_fills": broker.orders_filled,
            "win_rate": sum(t["won"] for t in trades) / n,
            "net_return": net_return,
            # total compounded return on initial capital, as a percent (final = start + net_return).
            "net_return_pct": net_return / broker.starting_balance * 100.0,
            "avg_pnl": net_return / n,
            # Total P&L at a flat 1 contract/trade: Σ(pnl_i / contracts_i). Unlike
            # net_return it is sizing/compounding-independent (each trade normalized
            # to one contract), yet frequency-aware (a sum, not a mean) — the metric
            # v2 ranks/optimizes on, so signal edge is scored free of Kelly leverage.
            "flat_stake_ev": sum(t["pnl"] / t["contracts"] for t in trades if t["contracts"]),
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            # Calmar = return / max drawdown. A monotonic-up curve has no drawdown
            # (impossible with real win/lose trades) — guarded to 0.0.
            "calmar": net_return / max_dd if max_dd > 0 else 0.0,
            "min_equity": float(np.min(equity)),
            "fill_rate": broker.orders_filled / broker.orders_submitted if broker.orders_submitted else 0.0,
            # `paid` (what the side actually cost), not the YES-denominated `entry_price`
            # — averaging that across a mixed yes/no book collapses to ~0.50 regardless
            # of what was really paid. `.get` keeps pre-existing trade dicts loadable.
            "avg_entry_price": sum(t.get("paid", t["entry_price"]) for t in trades) / n,
        }
