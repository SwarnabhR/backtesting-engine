"""
engine/regime_sizer.py
----------------------
Regime-aware portfolio allocation sizers.

Each sizer wraps a ``regime_fn`` callable that returns the current market
regime for a given symbol::

    regime_fn(df: pd.DataFrame, bar_idx: int) -> int   # +1 bull, -1 bear

Symbols in a bear regime have their weight set to zero; the remaining
(bull) symbols are re-weighted among themselves according to the sizer's
allocation logic.  This keeps gross exposure proportional to the number
of symbols that are actually in a favourable regime.

Built-in regime helpers
-----------------------
``slope_regime(df, bar_idx, period)``
    Returns +1 if the ``period``-bar linear regression slope of close
    prices is positive, -1 otherwise.  A pure price-momentum regime.

``supertrend_regime(df, bar_idx, period, multiplier)``
    Returns the Supertrend direction (+1 / -1) at ``bar_idx``.

Classes
-------
``RegimeEqualSizer``
    Zeroes bears, equal-weights survivors.  Optional per-symbol cap.

``RegimeVolSizer``
    Zeroes bears, inverse-volatility weights surviving symbols.
    Same ``lookback`` / ``rebalance_bars`` / ``cap`` knobs as
    :class:`~portfolio.VolTargetSizer`.

``RegimeFixedSizer``
    User-supplied static weights; zeros out bear symbols each bar.
    Weights are NOT renormalised after zeroing — the unallocated
    fraction sits in cash.

Usage
-----
    from regime_sizer import RegimeEqualSizer, slope_regime
    from portfolio import PortfolioBacktest
    from strategy import EMACrossoverRegime
    from functools import partial

    regime = partial(slope_regime, period=50)

    pb = PortfolioBacktest(
        initial=500_000,
        portfolio_sizer=RegimeEqualSizer(regime_fn=regime),
    )
    result = pb.run(dfs, EMACrossoverRegime)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import PortfolioSizer


# ============================================================================
# Built-in regime functions
# ============================================================================

def slope_regime(
    df: pd.DataFrame,
    bar_idx: int,
    period: int = 50,
) -> int:
    """
    Linear-regression-slope regime.

    Fits a least-squares line to the last ``period`` close prices up to
    and including ``bar_idx``.  Returns +1 if slope > 0, -1 otherwise.

    Parameters
    ----------
    df      : OHLCV DataFrame
    bar_idx : current bar index (integer position)
    period  : look-back window for the regression
    """
    start = max(0, bar_idx - period + 1)
    close = df["close"].iloc[start : bar_idx + 1].values
    if len(close) < 2:
        return 1  # not enough data — assume bull
    x = np.arange(len(close), dtype=float)
    # slope via normal equations
    x_mean = x.mean()
    y_mean = close.mean()
    slope = np.dot(x - x_mean, close - y_mean) / (np.dot(x - x_mean, x - x_mean) + 1e-12)
    return 1 if slope >= 0 else -1


def supertrend_regime(
    df: pd.DataFrame,
    bar_idx: int,
    period: int = 10,
    multiplier: float = 3.0,
) -> int:
    """
    Supertrend-based regime.

    Returns the Supertrend direction at ``bar_idx``:
    +1 (price above the band → bull), -1 (price below → bear).

    Parameters
    ----------
    df          : OHLCV DataFrame (must contain high, low, close)
    bar_idx     : current bar index (integer position)
    period      : ATR period for Supertrend
    multiplier  : ATR multiplier
    """
    # lazy import to avoid circular deps
    from indicators import supertrend as _supertrend
    # Compute on the slice up to bar_idx inclusive for efficiency
    end = bar_idx + 1
    slice_df = df.iloc[max(0, end - period * 4) : end]
    if len(slice_df) < period + 1:
        return 1
    _, direction = _supertrend(slice_df, period=period, multiplier=multiplier)
    return int(direction.iloc[-1])


# ============================================================================
# Regime-aware sizers
# ============================================================================

class RegimeEqualSizer(PortfolioSizer):
    """
    Equal-weight across *bull* symbols; bear symbols get zero weight.

    Parameters
    ----------
    regime_fn : callable(df, bar_idx) -> int
        Returns +1 (bull) or -1 (bear) for a given symbol at a given bar.
    cap : float
        Maximum weight any single symbol may receive.  Default 1.0 (no cap).
    """

    def __init__(self, regime_fn, cap: float = 1.0):
        self.regime_fn = regime_fn
        self.cap = cap

    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        if not symbols:
            return {}

        bull = [
            s for s in symbols
            if self.regime_fn(dfs[s], bar_idx) == 1
        ]

        if not bull:
            return {s: 0.0 for s in symbols}

        raw = 1.0 / len(bull)
        w   = min(raw, self.cap)
        return {s: (w if s in bull else 0.0) for s in symbols}


class RegimeVolSizer(PortfolioSizer):
    """
    Inverse-volatility weights restricted to *bull* symbols.

    Bear symbols get zero weight; survivors are weighted by 1/vol and
    capped at ``cap``.  Recomputed every ``rebalance_bars`` bars.

    Parameters
    ----------
    regime_fn      : callable(df, bar_idx) -> int
    lookback       : rolling volatility window (bars)
    rebalance_bars : how often to recompute weights
    cap            : per-symbol weight cap
    """

    def __init__(
        self,
        regime_fn,
        lookback: int = 20,
        rebalance_bars: int = 21,
        cap: float = 0.40,
    ):
        self.regime_fn       = regime_fn
        self.lookback        = lookback
        self.rebalance_bars  = rebalance_bars
        self.cap             = cap
        self._cache: dict[str, float] = {}
        self._last_rebalance = -1

    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        if not symbols:
            return {}

        # Identify bull symbols at this bar
        bull = [
            s for s in symbols
            if self.regime_fn(dfs[s], bar_idx) == 1
        ]

        # Rebalance if enough bars have passed or cache is stale
        if (
            bar_idx - self._last_rebalance >= self.rebalance_bars
            or not self._cache
        ):
            self._rebalance(symbols, bar_idx, dfs)
            self._last_rebalance = bar_idx

        # Zero out bears; return cached weights for bulls
        return {
            s: (self._cache.get(s, 0.0) if s in bull else 0.0)
            for s in symbols
        }

    def _rebalance(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
    ) -> None:
        inv_vols: dict[str, float] = {}
        for s in symbols:
            start = max(0, bar_idx - self.lookback)
            close = dfs[s]["close"].iloc[start : bar_idx + 1]
            if len(close) < 2:
                inv_vols[s] = 0.0
                continue
            vol = close.pct_change().dropna().std()
            inv_vols[s] = (1.0 / vol) if vol > 1e-10 else 0.0

        total = sum(inv_vols.values())
        if total == 0:
            equal = 1.0 / len(symbols) if symbols else 0.0
            self._cache = {s: equal for s in symbols}
        else:
            self._cache = {
                s: min(v / total, self.cap) for s, v in inv_vols.items()
            }


class RegimeFixedSizer(PortfolioSizer):
    """
    Static user-supplied weights; bear symbols are zeroed each bar.

    The unallocated fraction (from zeroed symbols) stays in cash — weights
    are deliberately NOT renormalised so users can reason about gross
    exposure explicitly.

    Parameters
    ----------
    weights    : dict[symbol, fraction]
    regime_fn  : callable(df, bar_idx) -> int
    """

    def __init__(self, weights: dict[str, float], regime_fn):
        self._weights  = weights
        self.regime_fn = regime_fn

    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for s in symbols:
            regime = self.regime_fn(dfs[s], bar_idx)
            result[s] = self._weights.get(s, 0.0) if regime == 1 else 0.0
        return result
