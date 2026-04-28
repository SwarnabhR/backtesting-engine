from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class CostModel(ABC):
    """
    Base class for all transaction cost models.

    A cost model receives the gross fill price, number of shares, and
    the notional trade value, and returns the total cost in currency units
    for *one side* of a trade (entry OR exit).

    The engine calls it twice per completed trade:
        total_cost = model.cost(entry_price, shares) + model.cost(exit_price, shares)

    The returned cost is always >= 0 and is subtracted from P&L.
    """

    @abstractmethod
    def cost(self, price: float, shares: int) -> float:
        """
        Parameters
        ----------
        price  : float  – gross fill price per share.
        shares : int    – number of shares traded.

        Returns
        -------
        float  – total cost for this leg, in the same currency as *price*.
        """


# ---------------------------------------------------------------------------
# Zero cost  (default / benchmark)
# ---------------------------------------------------------------------------

class NoCost(CostModel):
    """Zero commission and zero slippage.  The frictionless baseline."""

    def cost(self, price: float, shares: int) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "NoCost()"


# ---------------------------------------------------------------------------
# Fixed commission per trade
# ---------------------------------------------------------------------------

class FixedCommission(CostModel):
    """
    Flat fee per executed order, regardless of size.

    Example
    -------
    Zerodha equity delivery: ₹0 brokerage.
    Zerodha intraday / F&O: ₹20 per order.

    Parameters
    ----------
    commission : float
        Fixed fee per order leg, e.g. 20.0 for ₹20.
    """

    def __init__(self, commission: float = 20.0) -> None:
        if commission < 0:
            raise ValueError(f"commission must be >= 0, got {commission}")
        self.commission = commission

    def cost(self, price: float, shares: int) -> float:
        return self.commission

    def __repr__(self) -> str:
        return f"FixedCommission(₹{self.commission:.2f})"


# ---------------------------------------------------------------------------
# Percentage commission (most retail brokers outside India)
# ---------------------------------------------------------------------------

class PercentCommission(CostModel):
    """
    Commission as a percentage of notional trade value.

    Example
    -------
    0.1 % round-trip = 0.05 % per leg.  Typical for US equity brokers
    before the zero-commission era, and still common internationally.

    Parameters
    ----------
    pct : float
        Commission rate per leg, e.g. 0.0005 for 0.05 %.
    min_cost : float
        Minimum commission per order (default 0).  Set to e.g. 1.0
        to model a "greater of pct or ₹1" schedule.
    """

    def __init__(self, pct: float = 0.0005, min_cost: float = 0.0) -> None:
        if not 0 <= pct <= 1:
            raise ValueError(f"pct must be in [0, 1], got {pct}")
        self.pct      = pct
        self.min_cost = min_cost

    def cost(self, price: float, shares: int) -> float:
        notional = price * shares
        return max(notional * self.pct, self.min_cost)

    def __repr__(self) -> str:
        return f"PercentCommission({self.pct:.4%}, min=₹{self.min_cost})"


# ---------------------------------------------------------------------------
# Slippage: fixed ticks
# ---------------------------------------------------------------------------

class TickSlippage(CostModel):
    """
    Model market-impact as a fixed number of ticks (price increments).

    On entry the fill is assumed to be *ticks* above the close (you pay
    the spread).  On exit, *ticks* below the close (you give the spread).
    The engine calls cost() for each leg; the returned value represents the
    adverse fill cost: ``ticks × tick_size × shares``.

    Parameters
    ----------
    ticks     : float  – number of ticks of slippage per side (default 1).
    tick_size : float  – minimum price increment (default 0.05 for NSE equities).
    """

    def __init__(self, ticks: float = 1.0, tick_size: float = 0.05) -> None:
        if ticks < 0:
            raise ValueError(f"ticks must be >= 0, got {ticks}")
        if tick_size <= 0:
            raise ValueError(f"tick_size must be > 0, got {tick_size}")
        self.ticks     = ticks
        self.tick_size = tick_size

    def cost(self, price: float, shares: int) -> float:
        return self.ticks * self.tick_size * shares

    def __repr__(self) -> str:
        return f"TickSlippage(ticks={self.ticks}, tick_size={self.tick_size})"


# ---------------------------------------------------------------------------
# Slippage: percentage of price
# ---------------------------------------------------------------------------

class PercentSlippage(CostModel):
    """
    Model market-impact as a fraction of the fill price.

    Appropriate when the bid-ask spread scales with price (common for
    high-priced indices or illiquid names).  Typical NSE equity spread:
    0.01 %–0.05 % of price per side.

    Parameters
    ----------
    pct : float  – slippage as a fraction of price per leg,
                   e.g. 0.0002 for 0.02 %.
    """

    def __init__(self, pct: float = 0.0002) -> None:
        if not 0 <= pct <= 1:
            raise ValueError(f"pct must be in [0, 1], got {pct}")
        self.pct = pct

    def cost(self, price: float, shares: int) -> float:
        return price * shares * self.pct

    def __repr__(self) -> str:
        return f"PercentSlippage({self.pct:.4%})"


# ---------------------------------------------------------------------------
# Composite: combine any number of cost models
# ---------------------------------------------------------------------------

class CompositeCost(CostModel):
    """
    Sum the costs from multiple models.  This is the recommended way to
    combine commission + slippage.

    Example
    -------
    # Zerodha intraday: ₹20 flat + 0.02% slippage
    cost = CompositeCost(
        FixedCommission(20),
        PercentSlippage(0.0002),
    )

    # US retail: 0.05% commission + 1-tick slippage
    cost = CompositeCost(
        PercentCommission(0.0005),
        TickSlippage(ticks=1, tick_size=0.01),
    )
    """

    def __init__(self, *models: CostModel) -> None:
        if not models:
            raise ValueError("CompositeCost requires at least one model")
        self.models = list(models)

    def cost(self, price: float, shares: int) -> float:
        return sum(m.cost(price, shares) for m in self.models)

    def __repr__(self) -> str:
        parts = " + ".join(repr(m) for m in self.models)
        return f"CompositeCost({parts})"


# ---------------------------------------------------------------------------
# NSE convenience preset
# ---------------------------------------------------------------------------

def nse_equity_intraday() -> CompositeCost:
    """
    Approximate NSE equity intraday cost model:
      - Zerodha brokerage: ₹20 per order
      - STT (sell-side): 0.025 % of turnover  (absorbed into slippage approx)
      - Exchange + SEBI charges: ~0.005 %
      - Bid-ask spread: 0.02 % per side

    Total per-leg approximation: ₹20 flat + 0.025% percent slippage.
    """
    return CompositeCost(
        FixedCommission(20.0),
        PercentSlippage(0.00025),
    )


def nse_equity_delivery() -> CompositeCost:
    """
    NSE equity delivery (Zerodha):
      - ₹0 brokerage
      - STT buy+sell: 0.1 % each side
      - Exchange + SEBI charges: ~0.005 %
      - Bid-ask: 0.02 %

    Total per-leg: ~0.125 % percent slippage.
    """
    return CompositeCost(
        NoCost(),                    # zero brokerage
        PercentSlippage(0.00125),    # STT + exchange + spread
    )
