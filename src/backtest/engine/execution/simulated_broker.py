import copy
import logging
import math
from dataclasses import dataclass
from typing import Any

from src.backtest.engine.context import AccountSnapshot
from src.backtest.engine.execution.account import BinaryAccount
from src.backtest.engine.execution.fees import FeeModel, KalshiFees
from src.backtest.engine.execution.fills import CandleFills, FillModel
from src.backtest.engine.execution.lots import LotPolicy, TenthLots
from src.backtest.engine.execution.settlement import BinarySettlement, SettlementModel
from src.backtest.engine.feed import Feed
from src.backtest.engine.orders import Fill, Order

logger = logging.getLogger(__name__)

_QUANTITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class _RestingOrder:
    order: Order
    had_fill: bool = False


class KalshiBroker:
    """Runs orders against Kalshi's binary markets, delegating each rule to a policy.

    Owns the order lifecycle and the trade record; the account, lot size, fees,
    fills and settlement are the five objects it coordinates. Swapping any of them
    changes what the simulation models without touching this class or the engine.

    Keeps two feeds: the **master feed** supplies the quotes a fill is priced
    against, and the **market feed** supplies the outcome to settle on.

    Forks share their policies, so a policy must be stateless or every strategy in
    a batch will influence the others. Only the account is copied.
    """

    def __init__(
        self,
        starting_balance: float,
        master_feed: Feed,
        market_feed: Feed,
        id_col: str = "ticker",
        result_col: str = "result",
        account: BinaryAccount | None = None,
        lots: LotPolicy | None = None,
        fees: FeeModel | None = None,
        fills: FillModel | None = None,
        settlement: SettlementModel | None = None,
    ) -> None:
        self.starting_balance = starting_balance
        self.master_feed = master_feed
        self.market_feed = market_feed
        self.id_col = id_col
        self.result_col = result_col
        self._account = account or BinaryAccount(starting_balance)
        self.lots = lots or TenthLots()
        self.fees = fees or KalshiFees()
        self.fills = fills or CandleFills()
        self.settlement = settlement or BinarySettlement()

        self.trades: list[dict] = []
        self.orders_submitted = 0
        self.orders_filled = 0
        self._resting: list[_RestingOrder] = []
        self._last_id: Any = None
        self._finalized = False
        self._open_market()
        self._refresh_account()

    # --- engine-facing lifecycle ---------------------------------------------------

    def fork(self) -> "KalshiBroker":
        """A fresh broker with the same configuration, no state carried.

        Copies and resets rather than re-listing constructor arguments, so a policy
        added to ``__init__`` is not silently dropped here.
        """
        twin = copy.copy(self)
        twin._account = copy.copy(self._account)
        twin.reset()
        return twin

    def reset(self) -> None:
        """Clear all account state for a fresh run (keeps feeds and policies)."""
        self._account.reset()
        self.trades = []
        self.orders_submitted = 0
        self.orders_filled = 0
        self._resting = []
        self._last_id = None
        self._finalized = False
        self._open_market()
        self._refresh_account()

    def update(self) -> None:
        """Settle on a market boundary, then fill resting orders (1-candle delay).

        The only place the account can move, since ``receive_orders`` rests orders
        without touching cash, so refreshing ``account`` here covers every mutation.
        """
        if self._maybe_settle() | self._fill_resting():
            self._refresh_account()

    def receive_orders(self, orders: list[Order]) -> list[bool]:
        """Admit each order, or refuse it for size or affordability."""
        results: list[bool] = []
        for order in orders:
            self.orders_submitted += 1
            qty = self.lots.normalize(order.quantity)
            if qty == 0.0:
                results.append(False)
                continue
            if qty != order.quantity:
                order = Order(order.side, qty, order.order_type, order.limit_price)
            if order.order_type == "limit" and order.limit_price is None:
                results.append(False)
                continue
            if not self._can_afford(order):
                results.append(False)
                continue
            self._resting.append(_RestingOrder(order))
            results.append(True)
        return results

    def finalize(self) -> None:
        """Close the run. Idempotent; called once after the last tick.

        A no-op by design: the final market stays unsettled.
        """
        self._finalized = True

    def get_trades(self) -> list[dict]:
        """The settled-market trade records (a copy, so consumers can't mutate state)."""
        return list(self.trades)

    # --- account views -------------------------------------------------------------

    @property
    def balance(self) -> float:
        return self._account.balance

    @property
    def yes_contracts(self) -> float:
        return self._account.yes_contracts

    @property
    def no_contracts(self) -> float:
        return self._account.no_contracts

    def _refresh_account(self) -> None:
        self.account: AccountSnapshot = self._account.snapshot()

    # --- admission -----------------------------------------------------------------

    def _can_afford(self, order: Order) -> bool:
        """Check the fill model's admission estimate against available cash.

        Actual execution happens on a later tick and is checked again at its real
        price. An order is accepted when the model cannot yet provide an estimate.
        """
        fill = self.fills.estimate(order, self.master_feed)
        if fill is None:
            return True
        self._validate_fill(order, fill)
        return self._can_apply_fill(order, fill)

    # --- settlement ----------------------------------------------------------------

    def _maybe_settle(self) -> bool:
        """``True`` when a market boundary moved the account."""
        if len(self.market_feed) == 0:
            return False
        current_id = getattr(self.market_feed, self.id_col)[0]
        if current_id == self._last_id:
            return False
        settled = False
        if self._last_id is not None:
            # The just-closed market is now at [-1] with its outcome revealed.
            result = getattr(self.market_feed, self.result_col)[-1]
            self._settle(result, self.market_feed.close_time[-1])
            settled = True
        self._last_id = current_id
        self._open_market()
        return settled

    def _settle(self, result: Any, close_time: Any) -> None:
        payout = self.settlement.payout(result, self.yes_contracts, self.no_contracts)
        if payout is None:
            if self._mkt_contracts:
                logger.warning("Market settled with unknown result=%r; no payout", result)
        else:
            self._account.credit(payout)

        if self._mkt_contracts:
            self._record_trade(close_time)
        self._account.flatten()
        self._resting = []      # unfilled limit orders are cancelled at market close

    def _record_trade(self, close_time: Any) -> None:
        pnl = self.balance - self._mkt_start_balance
        self.trades.append({
            "close_time": close_time,
            "pnl": pnl,
            "won": pnl > 0,
            # YES-denominated, kept for backwards compatibility. For a NO position
            # this is 1 - (what we paid), so it is NOT the cost of the trade — use
            # `paid` for anything economic. Averaging it across a mixed yes/no book
            # collapses toward 0.50 by construction and means nothing.
            "entry_price": self._mkt_fill_notional / self._mkt_contracts,
            "paid": self._mkt_paid_notional / self._mkt_contracts,
            "side": self._mkt_side,             # "yes" | "no" | "mixed"
            "contracts": self._mkt_contracts,
        })

    def _open_market(self) -> None:
        self._mkt_start_balance = self.balance
        self._mkt_fill_notional = 0.0
        self._mkt_paid_notional = 0.0
        self._mkt_side: str | None = None
        self._mkt_contracts = 0.0

    # --- fills ---------------------------------------------------------------------

    def _fill_resting(self) -> bool:
        """``True`` when at least one order filled and moved the account."""
        if not self._resting:
            return False
        still_resting: list[_RestingOrder] = []
        filled = False
        for resting in self._resting:
            order = resting.order
            fill = self.fills.execute(order, self.master_feed)
            if fill is None:
                still_resting.append(resting)
                continue
            self._validate_fill(order, fill)
            filled_quantity = min(fill.quantity, order.quantity)
            filled_order = Order(order.side, filled_quantity, order.order_type, order.limit_price)
            if not self._apply_fill(filled_order, fill.price):
                continue

            if not resting.had_fill:
                self.orders_filled += 1
            filled = True
            remaining = order.quantity - filled_quantity
            if remaining > _QUANTITY_TOLERANCE:
                remainder = Order(order.side, remaining, order.order_type, order.limit_price)
                still_resting.append(_RestingOrder(remainder, had_fill=True))
        self._resting = still_resting
        return filled

    def _can_apply_fill(self, order: Order, fill: Fill) -> bool:
        fee = self.fees.charge(order.order_type, fill.price, fill.quantity)
        cost = self._account.cost_to_buy(order.side, fill.quantity, fill.price)
        return self._account.can_afford(fee + cost)

    @staticmethod
    def _validate_fill(order: Order, fill: Fill) -> None:
        overfilled = fill.quantity > order.quantity and not math.isclose(
            fill.quantity, order.quantity, rel_tol=0.0, abs_tol=_QUANTITY_TOLERANCE
        )
        if fill.quantity <= 0.0 or overfilled:
            raise ValueError(
                f"fill quantity must be in (0, {order.quantity}], got {fill.quantity}"
            )

    def _apply_fill(self, order: Order, p: float) -> bool:
        """Charge the fee and move the position, or refuse if it breaks the floor."""
        fee = self.fees.charge(order.order_type, p, order.quantity)
        cost = self._account.cost_to_buy(order.side, order.quantity, p)
        if not self._account.can_afford(fee + cost):
            return False
        self._account.charge(fee)
        self._account.buy(order.side, order.quantity, p)

        self._mkt_fill_notional += p * order.quantity
        # What this side actually costs per contract. NO pays 1-p, and the fee is
        # symmetric (0.07*p*(1-p)), so `paid` alone determines the economics.
        paid = p if order.side == "yes" else 1.0 - p
        self._mkt_paid_notional += paid * order.quantity
        self._mkt_side = (order.side if self._mkt_side in (None, order.side) else "mixed")
        self._mkt_contracts += order.quantity
        return True
