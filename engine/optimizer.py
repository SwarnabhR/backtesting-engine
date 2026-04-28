from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Type

import pandas as pd

from backtest import Backtest
from strategy import Strategy


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Holds the outcome of a single parameter combination."""
    params: dict[str, Any]
    sharpe_ratio: float
    cagr: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    total_return_pct: float

    def as_dict(self) -> dict:
        return {
            **self.params,
            "sharpe_ratio": self.sharpe_ratio,
            "cagr": self.cagr,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "total_return_pct": self.total_return_pct,
        }

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (
            f"OptimizationResult({p} | "
            f"sharpe={self.sharpe_ratio:.3f}, "
            f"cagr={self.cagr:.2%}, "
            f"dd={self.max_drawdown_pct:.2%}, "
            f"wr={self.win_rate:.2%}, "
            f"trades={self.total_trades})"
        )


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class GridOptimizer:
    """
    Brute-force grid-search optimizer.

    Parameters
    ----------
    strategy_class : Type[Strategy]
        Uninstantiated strategy class (e.g. ``EMACrossover``).
    param_grid : dict[str, list]
        Mapping of constructor argument names to lists of values to try.
        Example: ``{"fast": [5, 10, 12], "slow": [20, 26, 50]}``
    initial : float
        Starting capital passed to :class:`Backtest`.
    position_size : int
        Shares per trade passed to :class:`Backtest`.
    min_trades : int
        Discard results with fewer trades than this threshold (avoids
        over-fitted curves with only 1-2 lucky trades).
    constraint : Callable[[dict], bool] | None
        Optional function that takes a param dict and returns True if the
        combination is valid.  Use this to enforce relationships between
        params (e.g. ``lambda p: p["fast"] < p["slow"]``).  Invalid
        combinations are skipped before any backtest is run.
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: dict[str, list],
        initial: float = 10_000,
        position_size: int = 1,
        min_trades: int = 3,
        constraint: Callable[[dict], bool] | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.initial = initial
        self.position_size = position_size
        self.min_trades = min_trades
        self.constraint = constraint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate every parameter combination and return a ranked DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data as returned by :func:`data.fetch`.
        sort_by : str
            Column to rank by. One of: ``"sharpe_ratio"``, ``"cagr"``,
            ``"total_return_pct"``, ``"win_rate"``.
        ascending : bool
            Sort direction.  Default ``False`` (best first).

        Returns
        -------
        pd.DataFrame
            One row per valid parameter combination, sorted by *sort_by*.
        """
        valid_sort = {"sharpe_ratio", "cagr", "total_return_pct", "win_rate"}
        if sort_by not in valid_sort:
            raise ValueError(f"sort_by must be one of {valid_sort}, got {sort_by!r}")

        combinations = self._build_combinations()
        results: list[OptimizationResult] = []

        for params in combinations:
            if self.constraint is not None and not self.constraint(params):
                continue
            result = self._evaluate(df, params)
            if result is not None:
                results.append(result)

        if not results:
            return pd.DataFrame()

        df_results = pd.DataFrame([r.as_dict() for r in results])
        df_results = df_results.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        return df_results

    def best(
        self,
        df: pd.DataFrame,
        sort_by: str = "sharpe_ratio",
    ) -> OptimizationResult | None:
        """
        Return only the single best :class:`OptimizationResult`.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        sort_by : str
            Metric used to select the winner.

        Returns
        -------
        OptimizationResult or None
            Best result, or ``None`` if no valid combination was found.
        """
        ranked = self.run(df, sort_by=sort_by, ascending=False)
        if ranked.empty:
            return None

        top = ranked.iloc[0]
        param_cols = list(self.param_grid.keys())
        params = {col: top[col] for col in param_cols}
        return OptimizationResult(
            params=params,
            sharpe_ratio=float(top["sharpe_ratio"]),
            cagr=float(top["cagr"]),
            max_drawdown_pct=float(top["max_drawdown_pct"]),
            win_rate=float(top["win_rate"]),
            total_trades=int(top["total_trades"]),
            total_return_pct=float(top["total_return_pct"]),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_combinations(self) -> list[dict[str, Any]]:
        """Cartesian product of param_grid values, cast to native Python types."""
        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        combos = []
        for combo in itertools.product(*value_lists):
            # Cast numpy scalars / floats-that-should-be-ints back to int/float
            # so strategy constructors receive the expected types.
            native = {}
            for k, v in zip(keys, combo):
                if isinstance(v, float) and v == int(v):
                    native[k] = int(v)
                else:
                    # covers numpy scalar types (np.int64, np.float64, etc.)
                    try:
                        native[k] = v.item()  # numpy scalar -> Python scalar
                    except AttributeError:
                        native[k] = v         # already a plain Python type
            combos.append(native)
        return combos

    def _evaluate(
        self, df: pd.DataFrame, params: dict[str, Any]
    ) -> OptimizationResult | None:
        """Run a single backtest; return None if the run is invalid."""
        try:
            strategy = self.strategy_class(**params)
        except TypeError as exc:
            raise TypeError(
                f"Could not instantiate {self.strategy_class.__name__} "
                f"with params {params}: {exc}"
            ) from exc

        bt = Backtest(initial=self.initial, position_size=self.position_size)
        try:
            trades_df, equity_curve, metrics = bt.run(df, strategy)
        except Exception:
            # Silently skip parameter combos that cause runtime errors
            # (e.g., slow <= fast for EMA, degenerate edge cases).
            return None

        if metrics.total_trades < self.min_trades:
            return None

        return OptimizationResult(
            params=params,
            sharpe_ratio=float(metrics.sharpe_ratio),
            cagr=float(metrics.cagr),
            max_drawdown_pct=float(metrics.max_drawdown_pct),
            win_rate=float(metrics.win_rate),
            total_trades=int(metrics.total_trades),
            total_return_pct=float(metrics.total_return_pct),
        )
