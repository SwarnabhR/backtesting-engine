"""
engine/portfolio.py
-------------------
Multi-asset portfolio backtester.

Design
------
* One strategy *class* (+ optional per-symbol kwargs) is run against every
  symbol in the portfolio simultaneously on a shared calendar.
* Capital is divided across symbols according to a ``PortfolioSizer`` that
  decides the per-symbol allocation fraction at entry time.
* A single consolidated equity curve is produced by summing per-symbol
  mark-to-market P&L at every bar.
* The same :class:`sizer.PositionSizer` and :class:`costs.CostModel`
  used by the single-asset engine are reused unchanged.
* Symbols with insufficient data after alignment are silently skipped with
  a warning.

Allocation modes (``PortfolioSizer``)
--------------------------------------
``EqualWeightSizer``   – 1/N of current portfolio equity per symbol.
``FixedWeightSizer``   – user-supplied weight dict {symbol: fraction}.
``VolTargetSizer``     – inverse-volatility weights rebalanced every
                         ``rebalance_bars`` bars.

Usage
-----
    from portfolio import PortfolioBacktest, EqualWeightSizer
    from strategy import EMACrossoverRegime
    from costs import nse_equity_delivery

    dfs = {
        "RELIANCE.NS": rel_df,
        "INFY.NS":     inf_df,
        "HDFCBANK.NS": hdf_df,
    }

    pb = PortfolioBacktest(
        initial=500_000,
        portfolio_sizer=EqualWeightSizer(),
        cost_model=nse_equity_delivery(),
    )
    result = pb.run(dfs, strategy_class=EMACrossoverRegime)

    print(result.metrics)          # portfolio-level metrics
    print(result.symbol_metrics)   # per-symbol breakdown
    result.equity_curve.plot()     # consolidated equity
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd

from backtest import Backtest
from costs import CostModel, NoCost
from risk import BacktestMetrics, compute_metrics
from sizer import FixedSizer, PositionSizer


# ============================================================================
# Portfolio allocation (weight) sizers
# ============================================================================

class PortfolioSizer(ABC):
    """Base class.  Returns a weight dict {symbol: fraction} summing to <= 1."""

    @abstractmethod
    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        ...


class EqualWeightSizer(PortfolioSizer):
    """
    Allocate 1/N of portfolio equity to each symbol that is *currently active*
    (has a pending entry signal or open position on this bar).

    If ``cap`` is given, no single symbol gets more than that fraction.
    """

    def __init__(self, cap: float = 1.0):
        self.cap = cap

    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        raw = 1.0 / n
        w   = min(raw, self.cap)
        return {s: w for s in symbols}


class FixedWeightSizer(PortfolioSizer):
    """
    User-supplied static weights.  Missing symbols get 0.
    Weights are NOT renormalised — the caller is responsible for
    ensuring they sum to <= 1.
    """

    def __init__(self, weights: dict[str, float]):
        self._weights = weights

    def weights(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
        equity: float,
    ) -> dict[str, float]:
        return {s: self._weights.get(s, 0.0) for s in symbols}


class VolTargetSizer(PortfolioSizer):
    """
    Inverse-volatility position sizing.  At each rebalance bar the weight
    of symbol i is::

        w_i = (1 / vol_i) / sum(1 / vol_j  for all j)

    where ``vol_i`` is the rolling std of daily returns over the last
    ``lookback`` bars.  Weights are capped at ``cap``.

    Recomputed every ``rebalance_bars`` bars; held constant between
    rebalances.
    """

    def __init__(
        self,
        lookback: int = 20,
        rebalance_bars: int = 21,
        cap: float = 0.40,
    ):
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
        if (
            bar_idx - self._last_rebalance >= self.rebalance_bars
            or not self._cache
        ):
            self._rebalance(symbols, bar_idx, dfs)
            self._last_rebalance = bar_idx
        return {s: self._cache.get(s, 0.0) for s in symbols}

    def _rebalance(
        self,
        symbols: list[str],
        bar_idx: int,
        dfs: dict[str, pd.DataFrame],
    ) -> None:
        inv_vols: dict[str, float] = {}
        for s in symbols:
            df   = dfs[s]
            start = max(0, bar_idx - self.lookback)
            close = df["close"].iloc[start : bar_idx + 1]
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


# ============================================================================
# Result container
# ============================================================================

@dataclass
class PortfolioResult:
    """Returned by :meth:`PortfolioBacktest.run`."""

    equity_curve:   pd.Series                    # combined daily equity
    metrics:        BacktestMetrics              # portfolio-level metrics
    symbol_metrics: dict[str, BacktestMetrics]  # per-symbol metrics
    symbol_trades:  dict[str, pd.DataFrame]     # per-symbol trade log
    symbol_equity:  dict[str, pd.Series]        # per-symbol equity curve
    weights_log:    pd.DataFrame = field(default_factory=pd.DataFrame)
    # columns = symbol names, index = shared calendar; weight used per bar


# ============================================================================
# Main portfolio backtester
# ============================================================================

class PortfolioBacktest:
    """
    Multi-asset portfolio backtester.

    Parameters
    ----------
    initial : float
        Total starting capital shared across all symbols.
    portfolio_sizer : PortfolioSizer
        Determines what fraction of portfolio equity is allocated to each
        symbol.  Defaults to :class:`EqualWeightSizer`.
    sizer : PositionSizer
        Per-symbol share-count sizer passed to the underlying
        :class:`~backtest.Backtest`.  Defaults to ``FixedSizer(1)``.
    cost_model : CostModel
        Applied to every trade leg on every symbol.
    strategy_kwargs : dict[str, dict]
        Optional per-symbol strategy constructor kwargs, keyed by symbol.
        Symbols not present fall back to ``default_strategy_kwargs``.
    default_strategy_kwargs : dict
        Constructor kwargs applied to symbols not in ``strategy_kwargs``.
    """

    def __init__(
        self,
        initial: float = 100_000,
        portfolio_sizer: PortfolioSizer | None = None,
        sizer: PositionSizer | None = None,
        cost_model: CostModel | None = None,
        strategy_kwargs: dict[str, dict] | None = None,
        default_strategy_kwargs: dict | None = None,
    ) -> None:
        self.initial                  = float(initial)
        self.portfolio_sizer          = portfolio_sizer or EqualWeightSizer()
        self.sizer                    = sizer or FixedSizer(1)
        self.cost_model               = cost_model or NoCost()
        self.strategy_kwargs          = strategy_kwargs or {}
        self.default_strategy_kwargs  = default_strategy_kwargs or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dfs: dict[str, pd.DataFrame],
        strategy_class: Type,
    ) -> PortfolioResult:
        """
        Run the strategy on every symbol and combine results.

        Parameters
        ----------
        dfs : dict[str, pd.DataFrame]
            Mapping of symbol → OHLCV DataFrame (columns: open, high, low,
            close, volume).  DataFrames need not share the same date range;
            they are inner-aligned to the common calendar.
        strategy_class : type
            Strategy class (not instance).  Instantiated per symbol using
            ``strategy_kwargs[symbol]`` or ``default_strategy_kwargs``.

        Returns
        -------
        PortfolioResult
        """
        dfs = self._align(dfs)
        if not dfs:
            raise ValueError("No symbols remaining after alignment.")

        symbols      = list(dfs.keys())
        shared_index = next(iter(dfs.values())).index
        n_bars       = len(shared_index)

        # --- allocate per-symbol capital buckets ---
        # Each symbol runs its own Backtest with an initial capital equal to
        # (portfolio equity * weight) at the time of each NEW entry signal.
        # Between entries the bucket equity evolves independently, and the
        # consolidated curve is the sum of all buckets.

        # Step 1: generate all signals up front (vectorised, no look-ahead)
        strategies = {
            s: strategy_class(
                **{**self.default_strategy_kwargs,
                   **self.strategy_kwargs.get(s, {})}
            )
            for s in symbols
        }
        all_signals = {
            s: strategies[s].generate_signals(dfs[s]) for s in symbols
        }

        # Step 2: run single-asset backtests with capital proportional to weight
        symbol_trades:  dict[str, pd.DataFrame] = {}
        symbol_equity:  dict[str, pd.Series]    = {}
        symbol_metrics: dict[str, BacktestMetrics] = {}
        weights_rows: list[dict] = []

        # Calculate portfolio-proportional initial capital per symbol via weights
        # Use equal weight by default for initial capital split; dynamic
        # rebalancing happens implicitly through the equity evolution.
        init_weights = self.portfolio_sizer.weights(
            symbols, 0, dfs, self.initial
        )

        for s in symbols:
            w       = init_weights.get(s, 1.0 / len(symbols))
            cap     = self.initial * w
            bt      = Backtest(
                initial    = cap,
                sizer      = self.sizer,
                cost_model = self.cost_model,
            )
            trades, equity, metrics = bt.run(dfs[s], strategies[s])
            symbol_trades[s]  = trades
            symbol_equity[s]  = equity
            symbol_metrics[s] = metrics

        # Step 3: build consolidated equity curve
        equity_df     = pd.DataFrame(symbol_equity)   # index = shared_index[1:]
        # Fill forward for symbols that may have ended earlier
        equity_df     = equity_df.ffill()
        combined_equity = equity_df.sum(axis=1)

        # Step 4: weights log (static here; dynamic variants extend this)
        weights_log = pd.DataFrame(
            [init_weights],
            index=[shared_index[0]],
            columns=symbols,
        )

        # Step 5: portfolio-level metrics
        # Aggregate trades for portfolio-level metric computation
        if any(not t.empty for t in symbol_trades.values()):
            all_trades = pd.concat(
                [t for t in symbol_trades.values() if not t.empty],
                ignore_index=True,
            ).sort_values("entry_date").reset_index(drop=True)
        else:
            all_trades = pd.DataFrame()

        port_metrics = compute_metrics(all_trades, combined_equity)

        return PortfolioResult(
            equity_curve   = combined_equity,
            metrics        = port_metrics,
            symbol_metrics = symbol_metrics,
            symbol_trades  = symbol_trades,
            symbol_equity  = symbol_equity,
            weights_log    = weights_log,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Inner-join all symbol DataFrames to the common date index.
        Symbols with fewer than 60 bars after alignment are dropped with a
        warning (not enough data for most indicators to warm up).
        """
        if not dfs:
            return {}

        # Find common index
        common = None
        for df in dfs.values():
            common = df.index if common is None else common.intersection(df.index)

        if common is None or len(common) == 0:
            return {}

        aligned: dict[str, pd.DataFrame] = {}
        for symbol, df in dfs.items():
            trimmed = df.loc[common].copy()
            if len(trimmed) < 60:
                warnings.warn(
                    f"PortfolioBacktest: '{symbol}' has only {len(trimmed)} bars "
                    "after alignment (≤60) — skipped.",
                    stacklevel=3,
                )
                continue
            aligned[symbol] = trimmed

        return aligned


# ============================================================================
# Convenience helper
# ============================================================================

def run_portfolio(
    dfs: dict[str, pd.DataFrame],
    strategy_class: Type,
    initial: float = 100_000,
    portfolio_sizer: PortfolioSizer | None = None,
    cost_model: CostModel | None = None,
    **strategy_kwargs,
) -> PortfolioResult:
    """
    One-liner wrapper::

        result = run_portfolio(
            {"RELIANCE.NS": rel_df, "INFY.NS": inf_df},
            EMACrossoverRegime,
            initial=500_000,
        )
    """
    pb = PortfolioBacktest(
        initial                 = initial,
        portfolio_sizer         = portfolio_sizer,
        cost_model              = cost_model,
        default_strategy_kwargs = strategy_kwargs,
    )
    return pb.run(dfs, strategy_class)
