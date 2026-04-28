"""
Tests for engine/portfolio.py

Coverage
--------
* EqualWeightSizer / FixedWeightSizer / VolTargetSizer
* PortfolioBacktest.run() with 2 and 3 symbols
* Consolidated equity curve shape and ordering
* Per-symbol metrics + trades populated
* Alignment: short symbol dropped with a warning
* VolTargetSizer rebalancing and cap enforcement
* run_portfolio() convenience wrapper
* Empty / degenerate inputs
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from portfolio import (
    EqualWeightSizer,
    FixedWeightSizer,
    VolTargetSizer,
    PortfolioBacktest,
    PortfolioResult,
    run_portfolio,
)
from strategy import EMACrossover, EMACrossoverLS, EMACrossoverRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 300, seed: int = 42, trend: float = 0.0005) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame, ``n`` bars, starting 2020-01-01.
    ``trend`` controls the daily drift of the close price.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.001, 0.01, n))
    low   = close * (1 - rng.uniform(0.001, 0.01, n))
    op    = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


@pytest.fixture
def two_dfs():
    return {"A": _make_df(seed=1), "B": _make_df(seed=2)}


@pytest.fixture
def three_dfs():
    return {"X": _make_df(seed=10), "Y": _make_df(seed=20), "Z": _make_df(seed=30)}


# ---------------------------------------------------------------------------
# PortfolioSizer unit tests
# ---------------------------------------------------------------------------

class TestEqualWeightSizer:
    def test_two_symbols(self):
        s = EqualWeightSizer()
        w = s.weights(["A", "B"], 0, {}, 100_000)
        assert list(w.keys()) == ["A", "B"]
        assert abs(w["A"] - 0.5) < 1e-9
        assert abs(w["B"] - 0.5) < 1e-9

    def test_one_symbol(self):
        s = EqualWeightSizer()
        w = s.weights(["A"], 0, {}, 100_000)
        assert abs(w["A"] - 1.0) < 1e-9

    def test_cap_applied(self):
        s = EqualWeightSizer(cap=0.25)
        w = s.weights(["A", "B", "C"], 0, {}, 100_000)
        for v in w.values():
            assert v <= 0.25 + 1e-9

    def test_empty_returns_empty(self):
        s = EqualWeightSizer()
        assert s.weights([], 0, {}, 100_000) == {}


class TestFixedWeightSizer:
    def test_uses_provided_weights(self):
        s = FixedWeightSizer({"A": 0.6, "B": 0.3})
        w = s.weights(["A", "B"], 0, {}, 100_000)
        assert abs(w["A"] - 0.6) < 1e-9
        assert abs(w["B"] - 0.3) < 1e-9

    def test_missing_symbol_gets_zero(self):
        s = FixedWeightSizer({"A": 0.5})
        w = s.weights(["A", "B"], 0, {}, 100_000)
        assert w["B"] == 0.0


class TestVolTargetSizer:
    def test_returns_all_symbols(self, two_dfs):
        s   = VolTargetSizer(lookback=20, rebalance_bars=21)
        w   = s.weights(["A", "B"], 50, two_dfs, 100_000)
        assert set(w.keys()) == {"A", "B"}

    def test_weights_sum_le_one(self, three_dfs):
        s = VolTargetSizer(lookback=20)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        assert sum(w.values()) <= 1.0 + 1e-9

    def test_cap_enforced(self, three_dfs):
        s = VolTargetSizer(lookback=20, cap=0.30)
        w = s.weights(["X", "Y", "Z"], 50, three_dfs, 100_000)
        for v in w.values():
            assert v <= 0.30 + 1e-9

    def test_rebalances_after_interval(self, two_dfs):
        s = VolTargetSizer(rebalance_bars=5)
        w1 = s.weights(["A", "B"], 0,  two_dfs, 100_000)
        w2 = s.weights(["A", "B"], 3,  two_dfs, 100_000)  # within interval
        w3 = s.weights(["A", "B"], 10, two_dfs, 100_000)  # triggers rebalance
        # w1 and w2 should be identical (no rebalance)
        assert w1 == w2
        # w3 may differ (rebalanced at bar 10 with different vol window)
        # just verify it returns a valid dict
        assert set(w3.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# PortfolioBacktest integration tests
# ---------------------------------------------------------------------------

class TestPortfolioBacktest:
    def test_returns_portfolio_result(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result, PortfolioResult)

    def test_equity_curve_length(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        # equity curve has len(aligned_df) - 1 bars
        expected = len(_make_df()) - 1
        assert len(result.equity_curve) == expected

    def test_equity_curve_is_positive(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        assert (result.equity_curve > 0).all()

    def test_symbol_keys_present(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        assert set(result.symbol_metrics.keys()) == {"A", "B"}
        assert set(result.symbol_trades.keys())  == {"A", "B"}
        assert set(result.symbol_equity.keys())  == {"A", "B"}

    def test_portfolio_metrics_is_backtest_metrics(self, two_dfs):
        from risk import BacktestMetrics
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result.metrics, BacktestMetrics)

    def test_consolidated_equity_equals_sum_of_symbols(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossover)
        expected = sum(result.symbol_equity.values())
        pd.testing.assert_series_equal(
            result.equity_curve, expected, check_names=False, rtol=1e-6
        )

    def test_three_symbols(self, three_dfs):
        pb     = PortfolioBacktest(initial=300_000)
        result = pb.run(three_dfs, EMACrossover)
        assert set(result.symbol_equity.keys()) == {"X", "Y", "Z"}

    def test_long_short_strategy(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossoverLS)
        assert isinstance(result.equity_curve, pd.Series)

    def test_regime_filtered_strategy(self, two_dfs):
        pb     = PortfolioBacktest(initial=100_000)
        result = pb.run(two_dfs, EMACrossoverRegime)
        assert isinstance(result.equity_curve, pd.Series)

    def test_fixed_weight_sizer(self, two_dfs):
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=FixedWeightSizer({"A": 0.7, "B": 0.3}),
        )
        result = pb.run(two_dfs, EMACrossover)
        # A gets 70% of 100k = 70k, B gets 30k
        a_initial = result.symbol_equity["A"].iloc[0]
        b_initial = result.symbol_equity["B"].iloc[0]
        assert a_initial > b_initial

    def test_vol_target_sizer(self, two_dfs):
        pb = PortfolioBacktest(
            initial=100_000,
            portfolio_sizer=VolTargetSizer(lookback=20),
        )
        result = pb.run(two_dfs, EMACrossover)
        assert (result.equity_curve > 0).all()

    def test_per_symbol_kwargs(self, two_dfs):
        """Each symbol can receive different strategy params."""
        pb = PortfolioBacktest(
            initial=100_000,
            strategy_kwargs={
                "A": {"fast": 5,  "slow": 20},
                "B": {"fast": 10, "slow": 30},
            },
        )
        result = pb.run(two_dfs, EMACrossover)
        assert isinstance(result.equity_curve, pd.Series)


# ---------------------------------------------------------------------------
# Alignment / edge-case tests
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_short_symbol_dropped_with_warning(self):
        good_df  = _make_df(n=300, seed=1)
        short_df = _make_df(n=10,  seed=2)   # way too short
        dfs      = {"GOOD": good_df, "SHORT": short_df}
        pb       = PortfolioBacktest(initial=100_000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # short_df shares no dates with good_df (different seeds, same
            # start; actually they share the first 10 bdate rows)
            # Force non-overlapping by using a different start
            short_df2 = good_df.iloc[:10].copy()
            dfs2 = {"GOOD": good_df, "SHORT": short_df2}
            # After inner-join SHORT ends up with 10 bars (< 60) → warning
            result = pb.run(dfs2, EMACrossover)
        assert any("SHORT" in str(warning.message) for warning in w)
        # Only GOOD survives
        assert "SHORT" not in result.symbol_equity

    def test_no_common_dates_raises(self):
        df1 = _make_df(n=100, seed=1)
        # Shift df2 far into the future so no overlap
        df2 = _make_df(n=100, seed=2)
        df2.index = df2.index + pd.DateOffset(years=10)
        pb  = PortfolioBacktest(initial=100_000)
        with pytest.raises(ValueError, match="alignment"):
            pb.run({"A": df1, "B": df2}, EMACrossover)

    def test_empty_dfs_raises(self):
        pb = PortfolioBacktest(initial=100_000)
        with pytest.raises((ValueError, KeyError, Exception)):
            pb.run({}, EMACrossover)


# ---------------------------------------------------------------------------
# run_portfolio convenience wrapper
# ---------------------------------------------------------------------------

class TestRunPortfolio:
    def test_returns_portfolio_result(self, two_dfs):
        result = run_portfolio(two_dfs, EMACrossover, initial=100_000)
        assert isinstance(result, PortfolioResult)

    def test_strategy_kwargs_forwarded(self, two_dfs):
        result = run_portfolio(
            two_dfs, EMACrossover, initial=100_000, fast=5, slow=20
        )
        assert isinstance(result.equity_curve, pd.Series)
