from src.backtest.engine.context import AccountSnapshot

_MIN_BALANCE = 0.50


class BinaryAccount:
    """Cash and the two contract counts, and the only place either one changes.

    Buying a side first closes whatever is open on the other, then opens the
    remainder, so ``min(yes, no) == 0`` always holds and a single order can flip the
    position. Prices are YES-denominated throughout, as the fill model returns them;
    a NO contract costs ``1 - p``.

    Cash is never allowed below :data:`_MIN_BALANCE`, which is what makes an order
    affordable or not.
    """

    def __init__(self, starting_balance: float) -> None:
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.yes_contracts = 0.0
        self.no_contracts = 0.0

    def reset(self) -> None:
        self.balance = self.starting_balance
        self.yes_contracts = 0.0
        self.no_contracts = 0.0

    def can_afford(self, cost: float) -> bool:
        """Whether paying ``cost`` leaves the balance above the floor."""
        return self.balance - cost >= _MIN_BALANCE

    def cost_to_buy(self, side: str, quantity: float, price: float) -> float:
        """Net cash a buy consumes, after closing any opposite position.

        Closing releases what that side is worth, so a flip can cost less than it
        looks, or even pay. Fees are not included; the broker adds them.
        """
        if side == "yes":
            closed = min(quantity, self.no_contracts)
            return (quantity - closed) * price - closed * (1.0 - price)
        closed = min(quantity, self.yes_contracts)
        return (quantity - closed) * (1.0 - price) - closed * price

    def buy(self, side: str, quantity: float, price: float) -> None:
        """Apply a fill: close the opposite side, open the remainder, move cash."""
        if side == "yes":
            closed = min(quantity, self.no_contracts)
            self.no_contracts -= closed
            self.balance += closed * (1.0 - price)
            opened = quantity - closed
            self.yes_contracts += opened
            self.balance -= opened * price
        else:
            closed = min(quantity, self.yes_contracts)
            self.yes_contracts -= closed
            self.balance += closed * price
            opened = quantity - closed
            self.no_contracts += opened
            self.balance -= opened * (1.0 - price)

    def charge(self, fee: float) -> None:
        self.balance -= fee

    def credit(self, amount: float) -> None:
        self.balance += amount

    def flatten(self) -> None:
        """Drop both positions, as settlement does once a market has paid out."""
        self.yes_contracts = 0.0
        self.no_contracts = 0.0

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            cash=self.balance,
            positions={"yes": self.yes_contracts, "no": self.no_contracts},
        )
