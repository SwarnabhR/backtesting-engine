"""
Tests for engine/regime_sizer.py

Coverage
--------
* slope_regime / supertrend_regime helpers return +1 / -1
* RegimeEqualSizer  — bull weights, bear zeroed, all-bear, cap, empty
* RegimeVolSizer    — bull weights, bear zeroed, sum<=1, cap, rebalance
* RegimeFixedSizer  — bull uses user weight, bear zeroed, missing symbol
* Integration       — PortfolioBacktest.run() with each regime sizer
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from regime_sizer import (
    slope_regime, supertrend_regime,
    RegimeEqualSizer, RegimeVolSizer, RegimeFixedSizer,
)
from portfolio import PortfolioBacktest, PortfolioResult
from strategy import EMACrossover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 300,
    seed: int = 42,
    trend: float = 0.0005,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.001, 0.01, n))
    low   = close * (1 - rng.uniform(0.001, 0.01, n))
    op    = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _bull_regime(df, bar_idx):
    """Always bull — returns +1 unconditionally."""
    return 1


def _bear_regime(df, bar_idx):
    """Always bear — returns -1 unconditionally."""
    return -1


def _alternating_regime(df, bar_idx):
    """Bull on even bars, bear on odd bars."""
    return 1 if bar_idx % 2 == 0 else -1


@pytest.fixture
def two_dfs():
    return {"A": _make_df(seed=1), "B": _make_df(seed=2)}


@pytest.fixture
def three_dfs():
    return {"X": _make_df(seed=10), "Y": _make_df(seed=20), "Z": _make_df(seed=30)}


# ---------------------------------------------------------------------------
# Built-in regime helpers
# ---------------------------------------------------------------------------

class TestSlopeRegime:
    def test_returns_plus_one_in_uptrend(self):
        df = _make_df(trend=0.005)   # strong uptrend
        assert slope_regime(df, bar_idx=200, period=50) == 1

    def test_returns_minus_one_in_downtrend(self):
        rng = np.random.default_rng(0)
        n   = 300
        close = 200.0 * np.cumprod(1 + rng.normal(-0.005, 0.005, n))
        df = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999,
             "close": close, "volume": np.ones(n)},
            index=pd.bdate_range("2020-01-01", periods=n),
        )
        assert slope_regime(df, bar_idx=250, period=50) == -1

    def test_short_window_returns_int(self):
        df  = _make_df(n=10)
        val = slope_regime(df, bar_idx=5, period=50)  # period > available bars
        assert val in (1, -1)

    def test_single_bar_returns_bull(self):
        df = _make_df(n=5)
        assert slope_regime(df, bar_idx=0, period=50) == 1


class TestSupertrendRegime:
    def test_returns_valid_direction(self):
        df  = _make_df(n=200)
        val = supertrend_regime(df, bar_idx=150)
        assert val in (1, -1)

    def test_uptrend_is_bull(self):
        df  = _make_df(trend=0.008, n=200)
        val = supertrend_regime(df, bar_idx=180, period=7, multiplier=2.0)
        assert val == 1

    def test_short_window_fallback(self):
        df  = _make_df(n=5)
        val = supertrend_regime(df, bar_idx=2, period=10)
        assert val == 1   # fallback = bull


# ---------------------------------------------------------------------------
# RegimeEqualSizer
# ---------------------------------------------------------------------------

class TestRegimeEqualSizer:
    def test_all_bull_equal_weights(self, two_dfs):
        s = RegimeEqualSizer(regime_fn=_bull_regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert abs(w["A"] - 0.5) < 1e-9
        assert abs(w["B"] - 0.5) < 1e-9

    def test_one_bear_gets_zero(self, two_dfs):
        # A is bear, B is bull
        def regime(df, bar_idx):
            return -1 if df is two_dfs["A"] else 1
        s = RegimeEqualSizer(regime_fn=regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["A"] == 0.0
        assert abs(w["B"] - 1.0) < 1e-9  # sole survivor gets full weight

    def test_all_bear_zero_weights(self, two_dfs):
        s = RegimeEqualSizer(regime_fn=_bear_regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["A"] == 0.0
        assert w["B"] == 0.0

    def test_cap_applied(self, three_dfs):
        s = RegimeEqualSizer(regime_fn=_bull_regime, cap=0.25)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        for v in w.values():
            assert v <= 0.25 + 1e-9

    def test_empty_symbols_returns_empty(self, two_dfs):
        s = RegimeEqualSizer(regime_fn=_bull_regime)
        assert s.weights([], 0, two_dfs, 100_000) == {}

    def test_three_bull_one_bear(self, three_dfs):
        bear_sym = "X"
        def regime(df, bar_idx):
            return -1 if df is three_dfs[bear_sym] else 1
        s = RegimeEqualSizer(regime_fn=regime)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        assert w["X"] == 0.0
        assert abs(w["Y"] - 0.5) < 1e-9
        assert abs(w["Z"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# RegimeVolSizer
# ---------------------------------------------------------------------------

class TestRegimeVolSizer:
    def test_all_bull_weights_sum_le_one(self, three_dfs):
        s = RegimeVolSizer(regime_fn=_bull_regime, lookback=20)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        assert sum(w.values()) <= 1.0 + 1e-9

    def test_bear_symbol_gets_zero(self, two_dfs):
        def regime(df, bar_idx):
            return -1 if df is two_dfs["A"] else 1
        s = RegimeVolSizer(regime_fn=regime, lookback=20)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["A"] == 0.0
        assert w["B"] > 0.0

    def test_all_bear_all_zeros(self, two_dfs):
        s = RegimeVolSizer(regime_fn=_bear_regime, lookback=20)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["A"] == 0.0
        assert w["B"] == 0.0

    def test_cap_enforced(self, three_dfs):
        s = RegimeVolSizer(regime_fn=_bull_regime, lookback=20, cap=0.30)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        for v in w.values():
            assert v <= 0.30 + 1e-9

    def test_rebalance_interval_respected(self, two_dfs):
        s  = RegimeVolSizer(regime_fn=_bull_regime, rebalance_bars=5)
        w1 = s.weights(["A", "B"], 0, two_dfs, 100_000)
        w2 = s.weights(["A", "B"], 3, two_dfs, 100_000)   # within interval
        w3 = s.weights(["A", "B"], 10, two_dfs, 100_000)  # triggers rebalance
        assert w1 == w2
        assert set(w3.keys()) == {"A", "B"}

    def test_returns_all_symbols(self, two_dfs):
        s = RegimeVolSizer(regime_fn=_bull_regime, lookback=20)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert set(w.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# RegimeFixedSizer
# ---------------------------------------------------------------------------

class TestRegimeFixedSizer:
    def test_bull_uses_user_weight(self, two_dfs):
        s = RegimeFixedSizer({"A": 0.6, "B": 0.3}, regime_fn=_bull_regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert abs(w["A"] - 0.6) < 1e-9
        assert abs(w["B"] - 0.3) < 1e-9

    def test_bear_symbol_zeroed(self, two_dfs):
        def regime(df, bar_idx):
            return -1 if df is two_dfs["A"] else 1
        s = RegimeFixedSizer({"A": 0.6, "B": 0.3}, regime_fn=regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["A"] == 0.0
        assert abs(w["B"] - 0.3) < 1e-9  # NOT renormalised

    def test_all_bear_all_zeros(self, two_dfs):
        s = RegimeFixedSizer({"A": 0.5, "B": 0.5}, regime_fn=_bear_regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert all(v == 0.0 for v in w.values())

    def test_missing_symbol_gets_zero_weight_when_bull(self, two_dfs):
        s = RegimeFixedSizer({"A": 0.5}, regime_fn=_bull_regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert w["B"] == 0.0

    def test_weights_not_renormalised_after_zeroing(self, two_dfs):
        # A bear → zeroed; B bull → 0.3 stays 0.3, NOT bumped to 1.0
        def regime(df, bar_idx):
            return -1 if df is two_dfs["A"] else 1
        s = RegimeFixedSizer({"A": 0.6, "B": 0.3}, regime_fn=regime)
        w = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert abs(w["B"] - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# Integration: PortfolioBacktest.run() with each regime sizer
# ---------------------------------------------------------------------------

class TestRegimeSizerIntegration:
    def test_regime_equal_sizer_runs(self, two_dfs):
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=RegimeEqualSizer(regime_fn=_bull_regime),
        )
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result, PortfolioResult)
        assert (result.equity_curve > 0).all()

    def test_regime_vol_sizer_runs(self, two_dfs):
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=RegimeVolSizer(
                regime_fn=_bull_regime, lookback=20
            ),
        )
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result, PortfolioResult)

    def test_regime_fixed_sizer_runs(self, two_dfs):
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=RegimeFixedSizer(
                {"A": 0.5, "B": 0.5}, regime_fn=_bull_regime
            ),
        )
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result, PortfolioResult)

    def test_slope_regime_fn_integration(self, two_dfs):
        regime = partial(slope_regime, period=50)
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=RegimeEqualSizer(regime_fn=regime),
        )
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result.equity_curve, pd.Series)

    def test_three_symbols_regime_vol(self, three_dfs):
        regime = partial(slope_regime, period=30)
        pb = PortfolioBacktest(
            initial=300_000,
            portfolio_sizer=RegimeVolSizer(
                regime_fn=regime, lookback=20, cap=0.50
            ),
        )
        result = pb.run(three_dfs, EMACrossover)
        assert set(result.symbol_equity.keys()) == {"X", "Y", "Z"}
