from __future__ import annotations

from dataclasses import dataclass, field
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
            "fold":             self.fold,
            "is_start":         self.in_sample_start.date(),
            "is_end":           self.in_sample_end.date(),
            "oos_start":        self.out_sample_start.date(),
            "oos_end":          self.out_sample_end.date(),
            **{f"params_{k}": v for k, v in self.best_params.items()},
            "is_sharpe":        round(self.in_sample.sharpe_ratio, 3),
            "is_cagr":          round(self.in_sample.cagr, 4),
            "oos_sharpe":       round(self.out_sample.sharpe_ratio, 3),
            "oos_cagr":         round(self.out_sample.cagr, 4),
            "oos_max_dd":       round(self.out_sample.max_drawdown_pct, 4),
            "oos_win_rate":     round(self.out_sample.win_rate, 4),
            "oos_trades":       self.out_sample.total_trades,
        }


@dataclass
class WalkForwardResult:
    folds:            list[FoldResult]
    oos_equity_curve: pd.Series           # stitched OOS equity curve
    oos_metrics:      BacktestMetrics     # metrics on the stitched curve
    summary:          pd.DataFrame        # one row per fold

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
    Anchored walk-forward validation.

    Splits *df* into sequential folds.  Each fold has an in-sample (IS)
    window and an out-of-sample (OOS) window immediately after it.

    Two window modes
    ----------------
    ``anchored=True``  (default)
        The IS window always starts at the very first bar and grows with
        each fold.  Fold 1: IS=[0..n], OOS=[n..n+step].  Fold 2:
        IS=[0..n+step], OOS=[n+step..n+2*step], etc.
        Suitable when more history is always better (trend strategies).

    ``anchored=False``  (rolling)
        The IS window is a fixed-length sliding window.  Fold 1:
        IS=[0..is_bars], OOS=[is_bars..is_bars+oos_bars].  Fold 2:
        IS=[oos_bars..is_bars+oos_bars], OOS=[is_bars+oos_bars..is_bars+2*oos_bars].
        Suitable when recent data is more relevant (mean-reversion strategies).

    Parameters
    ----------
    strategy_class : Type[Strategy]
        Uninstantiated strategy class.
    param_grid : dict[str, list]
        Parameter grid passed to :class:`GridOptimizer`.
    is_bars : int
        Number of bars in the in-sample window.
    oos_bars : int
        Number of bars in each out-of-sample window.
    anchored : bool
        Window mode (see above). Default True.
    sort_by : str
        Metric used to select the best IS params.
        One of ``"sharpe_ratio"``, ``"cagr"``, ``"total_return_pct"``, ``"win_rate"``.
    initial : float
        Starting capital.
    position_size : int
        Shares per trade.
    min_trades : int
        Minimum trades in IS window to accept a parameter set.
    constraint : Callable | None
        Optional constraint lambda forwarded to :class:`GridOptimizer`.
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: dict[str, list],
        is_bars: int = 504,        # ~2 trading years
        oos_bars: int = 126,       # ~6 trading months
        anchored: bool = True,
        sort_by: str = "sharpe_ratio",
        initial: float = 100_000,
        position_size: int = 1,
        min_trades: int = 3,
        constraint: Callable[[dict], bool] | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.is_bars = is_bars
        self.oos_bars = oos_bars
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
        Run walk-forward validation on *df*.

        Returns
        -------
        WalkForwardResult
            Contains per-fold detail, stitched OOS equity curve, and
            aggregate OOS metrics.
        """
        folds = self._build_folds(df)
        if not folds:
            raise ValueError(
                f"Not enough data for even one fold. "
                f"Need at least {self.is_bars + self.oos_bars} bars, "
                f"got {len(df)}."
            )

        fold_results: list[FoldResult] = []
        oos_equity_segments: list[pd.Series] = []
        all_trades: list[pd.DataFrame] = []
        running_capital = self.initial

        for fold_num, (is_slice, oos_slice) in enumerate(folds, start=1):
            # 1. Optimise on IS window
            best_params = self._optimise(is_slice)
            if best_params is None:
                continue  # skip fold if grid yielded no valid results

            # 2. Evaluate IS metrics (for reporting only)
            is_metrics = self._evaluate(is_slice, best_params)

            # 3. Run OOS with best IS params, capital carries over between folds
            bt = Backtest(initial=running_capital, position_size=self.position_size)
            strategy = self.strategy_class(**best_params)
            oos_trades, oos_equity, oos_metrics = bt.run(oos_slice, strategy)

            # Update running capital for next fold
            if not oos_equity.empty:
                running_capital = float(oos_equity.iloc[-1])

            oos_equity_segments.append(oos_equity)
            all_trades.append(oos_trades)

            fold_results.append(FoldResult(
                fold=fold_num,
                in_sample_start=is_slice.index[0],
                in_sample_end=is_slice.index[-1],
                out_sample_start=oos_slice.index[0],
                out_sample_end=oos_slice.index[-1],
                best_params=best_params,
                in_sample=is_metrics,
                out_sample=oos_metrics,
            ))

        if not fold_results:
            raise ValueError("All folds failed — try reducing min_trades or expanding the param grid.")

        # Stitch OOS equity curve (re-base each segment to end of previous)
        stitched = self._stitch_equity(oos_equity_segments)

        # Compute aggregate metrics on stitched curve
        all_trades_df = pd.concat(
            [t for t in all_trades if not t.empty], ignore_index=True
        ) if any(not t.empty for t in all_trades) else pd.DataFrame()
        aggregate_metrics = compute_metrics(all_trades_df, stitched)

        summary = pd.DataFrame([f.as_dict() for f in fold_results])

        return WalkForwardResult(
            folds=fold_results,
            oos_equity_curve=stitched,
            oos_metrics=aggregate_metrics,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return list of (in_sample_df, oos_df) slices."""
        folds = []
        n = len(df)
        start = 0

        while True:
            if self.anchored:
                is_end = self.is_bars + (len(folds) * self.oos_bars)
            else:
                is_start = len(folds) * self.oos_bars
                is_end   = is_start + self.is_bars
                start    = is_start

            oos_end = is_end + self.oos_bars

            if oos_end > n:
                break

            is_slice  = df.iloc[start:is_end]
            oos_slice = df.iloc[is_end:oos_end]
            folds.append((is_slice, oos_slice))

        return folds

    def _optimise(self, is_df: pd.DataFrame) -> dict[str, Any] | None:
        """Run grid search on IS slice and return best params dict."""
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
        """Run a single backtest and return its metrics."""
        bt = Backtest(initial=self.initial, position_size=self.position_size)
        strategy = self.strategy_class(**params)
        _, _, metrics = bt.run(df, strategy)
        return metrics

    @staticmethod
    def _stitch_equity(segments: list[pd.Series]) -> pd.Series:
        """
        Concatenate equity segments so that each segment starts where
        the previous one ended (avoids jumps from capital carry-over).
        """
        if not segments:
            return pd.Series(dtype=float)
        return pd.concat([s for s in segments if not s.empty])
