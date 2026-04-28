"""Unit tests for engine/risk.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from risk import sharpe_ratio, cagr, max_drawdown, win_rate, total_return, compute_metrics


def _equity(values, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _trades(pnls):
    return pd.DataFrame({"pnl": pnls, "pnl_pct": [p / 100 for p in pnls]})


class TestSharpe:
    def test_positive_for_uptrend(self):
        eq = _equity(np.linspace(10_000, 15_000, 252))
        assert sharpe_ratio(eq) > 0

    def test_negative_for_downtrend(self):
        eq = _equity(np.linspace(15_000, 10_000, 252))
        assert sharpe_ratio(eq) < 0

    def test_flat_returns_zero(self):
        eq = _equity([10_000.0] * 100)
        assert sharpe_ratio(eq) == 0.0

    def test_single_bar_returns_zero(self):
        assert sharpe_ratio(_equity([10_000.0])) == 0.0


class TestCAGR:
    def test_known_value(self):
        """Double in 1 year → CAGR = 100%."""
        idx = pd.bdate_range("2020-01-02", periods=252)
        eq  = pd.Series([10_000.0] * 252, index=idx)
        eq.iloc[-1] = 20_000.0
        result = cagr(eq)
        # ~100% over ~252 business days (~1 year)
        assert 0.90 < result < 1.10

    def test_no_growth_zero(self):
        eq = _equity([10_000.0] * 252)
        assert abs(cagr(eq)) < 0.01

    def test_single_bar_zero(self):
        assert cagr(_equity([10_000.0])) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        """Peak 100, trough 50 → drawdown = -50%."""
        prices = [100, 90, 80, 70, 60, 50, 60, 70, 80, 90]
        dd, _ = max_drawdown(_equity(prices))
        assert abs(dd - (-0.50)) < 0.01

    def test_monotone_up_zero_dd(self):
        dd, _ = max_drawdown(_equity(np.linspace(100, 200, 50)))
        assert dd == 0.0 or abs(dd) < 1e-9

    def test_single_bar(self):
        dd, dur = max_drawdown(_equity([10_000.0]))
        assert dd == 0.0
        assert dur == 0


class TestWinRate:
    def test_all_wins(self):
        assert win_rate(_trades([10, 20, 5])) == 1.0

    def test_all_losses(self):
        assert win_rate(_trades([-5, -10])) == 0.0

    def test_mixed(self):
        assert win_rate(_trades([10, -5, 10, -5])) == 0.5

    def test_empty(self):
        assert win_rate(pd.DataFrame(columns=["pnl"])) == 0.0


class TestTotalReturn:
    def test_known(self):
        eq = _equity([10_000, 11_000, 12_000, 15_000])
        assert abs(total_return(eq) - 0.5) < 1e-9

    def test_flat_zero(self):
        assert total_return(_equity([10_000.0] * 10)) == 0.0


class TestComputeMetrics:
    def test_returns_dataclass(self):
        from risk import BacktestMetrics
        eq = _equity(np.linspace(10_000, 12_000, 252))
        m  = compute_metrics(_trades([50, 30, -10]), eq)
        assert isinstance(m, BacktestMetrics)
        assert m.total_trades == 3
        assert isinstance(m.sharpe_ratio, float)
