from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class BacktestMetrics:
    sharpe_ratio: float
    cagr: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate: float
    total_trades: int
    total_return_pct: float


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.05) -> float:
    daily_returns = equity_curve.pct_change().dropna()
    active_returns = daily_returns[daily_returns != 0]
    if len(active_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    excess = active_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve.iloc[0]
    end   = equity_curve.iloc[-1]
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if n_days <= 0 or start <= 0:
        return 0.0
    # Guard: end <= 0 means total ruin.  (end/start)**exp is undefined for
    # negative end; return -100% to signal complete capital loss.
    if end <= 0:
        return -1.0
    years = n_days / 365.25
    return float((end / start) ** (1.0 / years) - 1)


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd_pct = float(drawdown.min())

    underwater = drawdown < 0
    max_duration = 0
    current_duration = 0
    for is_under, _ in zip(underwater, equity_curve.items()):
        if is_under:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    if len(equity_curve) > 1 and max_duration > 1:
        avg_bar_days = (
            (equity_curve.index[-1] - equity_curve.index[0]).days
            / len(equity_curve)
        )
        duration_days = int(max_duration * avg_bar_days)
    else:
        duration_days = max_duration
    return max_dd_pct, duration_days


def win_rate(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    return float((trades_df["pnl"] > 0).sum() / len(trades_df))


def total_return(equity_curve: pd.Series) -> float:
    if equity_curve.iloc[0] == 0:
        return 0.0
    return float(
        (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    )


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.05,
) -> BacktestMetrics:
    dd_pct, dd_days = max_drawdown(equity_curve)
    return BacktestMetrics(
        sharpe_ratio=sharpe_ratio(equity_curve, risk_free_rate),
        cagr=cagr(equity_curve),
        max_drawdown_pct=dd_pct,
        max_drawdown_duration_days=dd_days,
        win_rate=win_rate(trades_df),
        total_trades=len(trades_df),
        total_return_pct=total_return(equity_curve),
    )
