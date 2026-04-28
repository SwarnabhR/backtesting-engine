from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PositionSizer(ABC):
    """
    Base class for all position sizers.

    Every sizer receives the current equity and the entry-bar row of OHLCV
    data, and returns the number of *whole shares* to trade.
    """

    @abstractmethod
    def size(self, equity: float, bar: pd.Series) -> int:
        """
        Parameters
        ----------
        equity : float
            Current portfolio equity at the moment of entry.
        bar : pd.Series
            The OHLCV row for the entry bar (index = column names).
            Use ``bar["close"]`` for the entry price,
            ``bar["atr"]`` if ATR has been pre-computed and attached.

        Returns
        -------
        int
            Number of whole shares (always >= 1).
        """


# ---------------------------------------------------------------------------
# Fixed share count  (legacy default — 1 share, unchanged behaviour)
# ---------------------------------------------------------------------------

class FixedSizer(PositionSizer):
    """
    Always trade exactly *shares* units.
    Equivalent to the old ``position_size=1`` default.
    """

    def __init__(self, shares: int = 1) -> None:
        if shares < 1:
            raise ValueError(f"shares must be >= 1, got {shares}")
        self.shares = shares

    def size(self, equity: float, bar: pd.Series) -> int:
        return self.shares

    def __repr__(self) -> str:
        return f"FixedSizer(shares={self.shares})"


# ---------------------------------------------------------------------------
# Percent-of-equity  (most common for retail backtests)
# ---------------------------------------------------------------------------

class PercentEquitySizer(PositionSizer):
    """
    Risk a fixed fraction of current equity per trade.

    shares = floor( equity * pct / close )

    Example
    -------
    equity=100_000, pct=0.02, close=18_000  →  floor(2000/18000) = 0 shares
    equity=100_000, pct=0.10, close=18_000  →  floor(10000/18000) = 0 shares

    Note: for high-priced indices (Nifty ~18 000) a 10 % allocation still
    buys 0 shares at 1-lot granularity.  Use a CFD/futures multiplier or
    switch to ``ATRSizer`` which sizes by risk, not notional value.

    Parameters
    ----------
    pct : float
        Fraction of equity to allocate, e.g. 0.10 for 10 %.
    min_shares : int
        Floor to guarantee at least this many shares (default 1).
    """

    def __init__(self, pct: float = 0.02, min_shares: int = 1) -> None:
        if not 0 < pct <= 1:
            raise ValueError(f"pct must be in (0, 1], got {pct}")
        self.pct = pct
        self.min_shares = min_shares

    def size(self, equity: float, bar: pd.Series) -> int:
        price = float(bar["close"])
        if price <= 0:
            return self.min_shares
        shares = int(equity * self.pct / price)
        return max(shares, self.min_shares)

    def __repr__(self) -> str:
        return f"PercentEquitySizer(pct={self.pct:.1%}, min_shares={self.min_shares})"


# ---------------------------------------------------------------------------
# ATR-based risk sizing  (professional standard)
# ---------------------------------------------------------------------------

class ATRSizer(PositionSizer):
    """
    Size positions so that one ATR of adverse move equals a fixed
    fraction of equity — the Van Tharp / Curtis Faith method.

    shares = floor( equity * risk_pct / (atr_mult * ATR) )

    The ATR must be pre-attached to the DataFrame as a column named
    ``"atr"`` before running the backtest.  Use
    ``indicators.atr(df, period)`` to compute it.

    Parameters
    ----------
    risk_pct : float
        Maximum fraction of equity to risk per trade, e.g. 0.01 (1 %).
    atr_mult : float
        Stop-loss width in ATR units (default 2.0).
        The dollar stop = atr_mult * ATR per share.
    atr_col : str
        Column name in the bar Series (default ``"atr"``).
    min_shares : int
        Floor shares (default 1).

    Example
    -------
    equity=100_000, risk_pct=0.01, ATR=150, atr_mult=2.0
    → risk_budget = 1_000
    → stop_per_share = 300
    → shares = floor(1_000 / 300) = 3
    """

    def __init__(
        self,
        risk_pct: float = 0.01,
        atr_mult: float = 2.0,
        atr_col: str = "atr",
        min_shares: int = 1,
    ) -> None:
        if not 0 < risk_pct <= 1:
            raise ValueError(f"risk_pct must be in (0, 1], got {risk_pct}")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be > 0, got {atr_mult}")
        self.risk_pct = risk_pct
        self.atr_mult = atr_mult
        self.atr_col = atr_col
        self.min_shares = min_shares

    def size(self, equity: float, bar: pd.Series) -> int:
        atr = float(bar.get(self.atr_col, 0))
        if atr <= 0:
            # ATR not available or zero — fall back to 1 share
            return self.min_shares
        stop_per_share = self.atr_mult * atr
        risk_budget = equity * self.risk_pct
        shares = int(risk_budget / stop_per_share)
        return max(shares, self.min_shares)

    def __repr__(self) -> str:
        return (
            f"ATRSizer(risk_pct={self.risk_pct:.1%}, "
            f"atr_mult={self.atr_mult}, atr_col='{self.atr_col}')"
        )


# ---------------------------------------------------------------------------
# Kelly fraction  (aggressive, theory-optimal, needs win/loss history)
# ---------------------------------------------------------------------------

class KellySizer(PositionSizer):
    """
    Size using the Kelly criterion:  f* = (bp - q) / b
    where
        b  = average win / average loss  (win-loss ratio)
        p  = historical win rate
        q  = 1 - p

    A *fractional Kelly* (e.g. half-Kelly) is strongly recommended to
    reduce variance.  Full Kelly maximises long-run geometric growth but
    causes extreme drawdowns in practice.

    Parameters
    ----------
    win_rate : float
        Historical win rate, e.g. 0.55.
    avg_win : float
        Average winning trade return (as a fraction, e.g. 0.03 for 3 %).
    avg_loss : float
        Average losing trade return (positive fraction, e.g. 0.015).
    fraction : float
        Kelly fraction to apply (default 0.5 = half-Kelly).
    min_shares : int
        Floor shares.
    """

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
        min_shares: int = 1,
    ) -> None:
        if not 0 < win_rate < 1:
            raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be > 0, got {avg_loss}")
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction
        self.min_shares = min_shares

        b = avg_win / avg_loss
        p, q = win_rate, 1 - win_rate
        raw_f = (b * p - q) / b
        self._kelly_f = max(0.0, raw_f) * fraction   # clamp to [0, fraction]

    @classmethod
    def from_trades(cls, trades_df, fraction: float = 0.5) -> "KellySizer":
        """
        Convenience constructor: derive win_rate, avg_win, avg_loss
        directly from a completed trades DataFrame.
        """
        if trades_df.empty:
            raise ValueError("trades_df is empty — cannot compute Kelly parameters")
        wins  = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
        losses = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].abs()
        win_rate  = len(wins) / len(trades_df)
        avg_win   = float(wins.mean())   if len(wins)   > 0 else 0.001
        avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.001
        return cls(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            fraction=fraction,
        )

    def size(self, equity: float, bar: pd.Series) -> int:
        price = float(bar["close"])
        if price <= 0 or self._kelly_f <= 0:
            return self.min_shares
        alloc = equity * self._kelly_f
        shares = int(alloc / price)
        return max(shares, self.min_shares)

    def __repr__(self) -> str:
        return (
            f"KellySizer(f*={self._kelly_f:.3f}, "
            f"wr={self.win_rate:.1%}, fraction={self.fraction})"
        )
