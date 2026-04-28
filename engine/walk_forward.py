from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type

import pandas as pd

from backtest import Backtest
from optimizer import GridOptimizer, OptimizationResult
from risk import compute_metrics, BacktestMetrics
from strategy import Strategy


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    in_sample_start:  pd.Timestamp
    in_sample_end:    pd.Timestamp
    out_sample_start: pd.Timestamp
    out_sample_end:   pd.Timestamp
    best_params:      dict[str, Any]
    in_sample:        BacktestMetrics
    out_sample:       BacktestMetrics

    def as_dict(self) -> dict:
        return {
            "fold":          self.fold,
            "is_start":      self.in_sample_start.date(),
            "is_end":        self.in_sample_end.date(),
            "oos_start":     self.out_sample_start.date(),
            "oos_end":       self.out_sample_end.date(),
            **{f"params_{k}": v for k, v in self.best_params.items()},
            "is_sharpe":     round(self.in_sample.sharpe_ratio, 3),
            "is_cagr":       round(self.in_sample.cagr, 4),
            "oos_sharpe":    round(self.out_sample.sharpe_ratio, 3),
            "oos_cagr":      round(self.out_sample.cagr, 4),
            "oos_max_dd":    round(self.out_sample.max_drawdown_pct, 4),
            "oos_win_rate":  round(self.out_sample.win_rate, 4),
            "oos_trades":    self.out_sample.total_trades,
        }


@dataclass
class WalkForwardResult:
    folds:            list[FoldResult]
    oos_equity_curve: pd.Series
    oos_metrics:      BacktestMetrics
    summary:          pd.DataFrame

    def __repr__(self) -> str:
        m = self.oos_metrics
        return (
            f"WalkForwardResult({len(self.folds)} folds | "
            f"OOS sharpe={m.sharpe_ratio:.3f}, "
            f"cagr={m.cagr:.2%}, "
            f"dd={m.max_drawdown_pct:.2%}, "
            f"wr={m.win_rate:.2%}, "
            f"trades={m.total_trades})"
        )


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Anchored or rolling walk-forward validation.

    Window modes
    ------------
    ``anchored=True``  (default)
        IS window always starts at bar 0 and grows each fold.
        Best for trend strategies where more history helps.

    ``anchored=False``  (rolling)
        IS window is a fixed-length sliding window.
        Best for mean-reversion where recent data matters more.

    Warmup handling
    ---------------
    Some indicators (e.g. 200-bar SMA for trend_regime) need a burn-in
    period before they emit valid values.  Set ``warmup_bars`` to the
    longest such lookback.  The IS window will be padded so that
    ``is_bars`` of *usable* bars remain after warmup is discarded.
    The OOS slice always starts immediately after the IS window; it reuses
    the tail of the IS window as its own warmup context, so no OOS bars
    are wasted.

    Parameters
    ----------
    strategy_class : Type[Strategy]
    param_grid : dict[str, list]
    is_bars : int
        Usable (post-warmup) IS bars per fold.
    oos_bars : int
        OOS bars per fold.
    warmup_bars : int
        Indicator warmup bars prepended to every IS slice (default 0).
        Set to your longest indicator lookback, e.g. 200 for regime filter.
    anchored : bool
    sort_by : str
    initial : float
    position_size : int
    min_trades : int
        Minimum IS trades to accept a param set. Default 2 (regime
        strategies are infrequent; 3 is too strict for short windows).
    constraint : Callable | None
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: dict[str, list],
        is_bars: int = 504,
        oos_bars: int = 252,        # default raised to ~1 yr: more OOS trades
        warmup_bars: int = 0,       # set to longest indicator lookback
        anchored: bool = True,
        sort_by: str = "sharpe_ratio",
        initial: float = 100_000,
        position_size: int = 1,
        min_trades: int = 2,        # lowered from 3; regime strategies are sparse
        constraint: Callable[[dict], bool] | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.is_bars = is_bars
        self.oos_bars = oos_bars
        self.warmup_bars = warmup_bars
        self.anchored = anchored
        self.sort_by = sort_by
        self.initial = initial
        self.position_size = position_size
        self.min_trades = min_trades
        self.constraint = constraint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Returns
        -------
        WalkForwardResult
        """
        folds = self._build_folds(df)
        if not folds:
            total_needed = self.warmup_bars + self.is_bars + self.oos_bars
            raise ValueError(
                f"Not enough data for even one fold. "
                f"Need at least {total_needed} bars "
                f"(warmup={self.warmup_bars} + IS={self.is_bars} + OOS={self.oos_bars}), "
                f"got {len(df)}."
            )

        fold_results: list[FoldResult] = []
        oos_equity_segments: list[pd.Series] = []
        all_trades: list[pd.DataFrame] = []
        running_capital = self.initial

        for fold_num, (is_slice, oos_slice) in enumerate(folds, start=1):
            # 1. Optimise on full IS slice (includes warmup prefix)
            best_params = self._optimise(is_slice)
            if best_params is None:
                print(f"  [fold {fold_num}] skipped — no param combo met min_trades={self.min_trades}")
                continue

            # 2. IS metrics (full slice with warmup; warmup bars produce no
            #    trades so they don't inflate IS numbers)
            is_metrics = self._evaluate(is_slice, best_params)

            # 3. OOS: prepend warmup prefix from tail of IS so indicators
            #    initialise correctly, then run backtest on OOS bars only.
            #    This is the key fix: without context the 200-bar SMA is NaN
            #    for the entire short OOS window.
            oos_with_context = self._oos_with_warmup(is_slice, oos_slice)
            bt = Backtest(initial=running_capital, position_size=self.position_size)
            strategy = self.strategy_class(**best_params)
            _, oos_equity_full, _ = bt.run(oos_with_context, strategy)

            # Trim equity curve back to OOS-only bars
            oos_equity = oos_equity_full.loc[oos_slice.index[0]:]
            # Re-run trades/metrics on OOS-only slice for clean reporting
            bt2 = Backtest(initial=running_capital, position_size=self.position_size)
            oos_trades, oos_equity2, oos_metrics = bt2.run(oos_slice, strategy)
            # Use the context-aware equity for the equity curve
            # but OOS-only metrics for the fold report
            if not oos_equity.empty:
                running_capital = float(oos_equity.iloc[-1])

            oos_equity_segments.append(oos_equity)
            all_trades.append(oos_trades)

            fold_results.append(FoldResult(
                fold=fold_num,
                in_sample_start=is_slice.index[self.warmup_bars] if self.warmup_bars else is_slice.index[0],
                in_sample_end=is_slice.index[-1],
                out_sample_start=oos_slice.index[0],
                out_sample_end=oos_slice.index[-1],
                best_params=best_params,
                in_sample=is_metrics,
                out_sample=oos_metrics,
            ))

        if not fold_results:
            raise ValueError(
                "All folds were skipped.  Try:\n"
                "  - Reducing min_trades (currently {self.min_trades})\n"
                "  - Expanding the param_grid\n"
                "  - Increasing is_bars or oos_bars"
            )

        stitched = self._stitch_equity(oos_equity_segments)
        all_trades_df = (
            pd.concat([t for t in all_trades if not t.empty], ignore_index=True)
            if any(not t.empty for t in all_trades)
            else pd.DataFrame()
        )
        aggregate_metrics = compute_metrics(all_trades_df, stitched)
        summary = pd.DataFrame([f.as_dict() for f in fold_results])

        return WalkForwardResult(
            folds=fold_results,
            oos_equity_curve=stitched,
            oos_metrics=aggregate_metrics,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Fold builder
    # ------------------------------------------------------------------

    def _build_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build (is_slice, oos_slice) pairs.

        is_slice always contains warmup_bars + is_bars rows so indicators
        have enough history.  oos_slice is purely the OOS window; the
        caller prepends warmup context separately.
        """
        folds = []
        n = len(df)
        total_is = self.warmup_bars + self.is_bars

        fold_idx = 0
        while True:
            if self.anchored:
                is_start = 0
                is_end   = total_is + fold_idx * self.oos_bars
            else:
                is_start = fold_idx * self.oos_bars
                is_end   = is_start + total_is

            oos_start = is_end
            oos_end   = oos_start + self.oos_bars

            if oos_end > n:
                break

            folds.append((df.iloc[is_start:is_end], df.iloc[oos_start:oos_end]))
            fold_idx += 1

        return folds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _oos_with_warmup(
        self, is_slice: pd.DataFrame, oos_slice: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepend the last ``warmup_bars`` rows of IS to OOS so that
        indicators initialise correctly on the OOS window.
        If warmup_bars == 0, returns oos_slice unchanged.
        """
        if self.warmup_bars == 0:
            return oos_slice
        context = is_slice.iloc[-self.warmup_bars:]
        return pd.concat([context, oos_slice])

    def _optimise(self, is_df: pd.DataFrame) -> dict[str, Any] | None:
        opt = GridOptimizer(
            strategy_class=self.strategy_class,
            param_grid=self.param_grid,
            initial=self.initial,
            position_size=self.position_size,
            min_trades=self.min_trades,
            constraint=self.constraint,
        )
        best: OptimizationResult | None = opt.best(is_df, sort_by=self.sort_by)
        return best.params if best is not None else None

    def _evaluate(
        self, df: pd.DataFrame, params: dict[str, Any]
    ) -> BacktestMetrics:
        bt = Backtest(initial=self.initial, position_size=self.position_size)
        strategy = self.strategy_class(**params)
        _, _, metrics = bt.run(df, strategy)
        return metrics

    @staticmethod
    def _stitch_equity(segments: list[pd.Series]) -> pd.Series:
        if not segments:
            return pd.Series(dtype=float)
        return pd.concat([s for s in segments if not s.empty])
