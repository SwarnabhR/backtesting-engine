"""Tests for the five new indicators and 15 new strategy classes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from indicators import supertrend, ichimoku, williams_r, donchian_channels, parabolic_sar
from strategy import (
    SupertrendStrategy, SupertrendLS, SupertrendRegime,
    IchimokuStrategy, IchimokuLS, IchimokuRegime,
    WilliamsRStrategy, WilliamsRLS, WilliamsRRegime,
    DonchianBreakout, DonchianBreakoutLS, DonchianBreakoutRegime,
    PSARStrategy, PSARLS, PSARRegime,
)


# ============================================================================
# Indicator unit tests
# ============================================================================

class TestSupertrend:
    def test_returns_two_series(self, large_df):
        st, direction = supertrend(large_df)
        assert isinstance(st, pd.Series) and isinstance(direction, pd.Series)

    def test_direction_values(self, large_df):
        _, direction = supertrend(large_df)
        assert set(direction.unique()).issubset({1, -1})

    def test_length(self, large_df):
        st, direction = supertrend(large_df)
        assert len(st) == len(large_df) == len(direction)

    def test_uptrend_direction(self, trending_up_df):
        """A clean uptrend should end with Supertrend direction = +1."""
        _, direction = supertrend(trending_up_df, period=10, multiplier=3.0)
        assert direction.iloc[-1] == 1

    def test_downtrend_direction(self, trending_down_df):
        _, direction = supertrend(trending_down_df, period=10, multiplier=3.0)
        assert direction.iloc[-1] == -1


class TestIchimoku:
    def test_returns_dict_with_five_keys(self, large_df):
        ic = ichimoku(large_df)
        assert set(ic.keys()) == {"tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou"}

    def test_lengths(self, large_df):
        ic = ichimoku(large_df)
        for v in ic.values():
            assert len(v) == len(large_df)

    def test_tenkan_faster_than_kijun(self, trending_up_df):
        """In an uptrend Tenkan (9) should be above Kijun (26) at the end."""
        ic = ichimoku(trending_up_df)
        t = ic["tenkan_sen"].dropna()
        k = ic["kijun_sen"].dropna()
        assert t.iloc[-1] >= k.iloc[-1]


class TestWilliamsR:
    def test_bounds(self, oscillating_df):
        wr = williams_r(oscillating_df)
        valid = wr.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_length(self, large_df):
        assert len(williams_r(large_df)) == len(large_df)

    def test_uptrend_less_negative(self, trending_up_df):
        """In a strong uptrend %R should be near 0 (overbought) at the end."""
        wr = williams_r(trending_up_df).dropna()
        assert wr.iloc[-1] > -50

    def test_downtrend_more_negative(self, trending_down_df):
        wr = williams_r(trending_down_df).dropna()
        assert wr.iloc[-1] < -50


class TestDonchianChannels:
    def test_returns_three_series(self, large_df):
        result = donchian_channels(large_df)
        assert len(result) == 3

    def test_upper_above_lower(self, oscillating_df):
        upper, _, lower = donchian_channels(oscillating_df)
        valid_u = upper.dropna(); valid_l = lower.dropna()
        assert (valid_u >= valid_l).all()

    def test_middle_is_midpoint(self, large_df):
        upper, middle, lower = donchian_channels(large_df, period=20)
        expected = (upper + lower) / 2
        pd.testing.assert_series_equal(middle, expected, check_names=False)

    def test_flat_zero_width(self, flat_df):
        """
        When high and low are identical across all bars the Donchian channel
        must have zero width (upper == lower) after the warmup period.

        NOTE: flat_df uses OHLC candles where high != close (the rolling max
        of *high* drives the upper band, not close), so we cannot assert
        upper == 100. The invariant we actually care about is that the channel
        collapses to zero width when prices are constant.
        """
        upper, _, lower = donchian_channels(flat_df, period=5)
        width = (upper - lower).iloc[10:]
        np.testing.assert_allclose(width.values, 0.0, atol=1e-8)


class TestParabolicSAR:
    def test_returns_two_series(self, large_df):
        sar, direction = parabolic_sar(large_df)
        assert isinstance(sar, pd.Series) and isinstance(direction, pd.Series)

    def test_direction_values(self, large_df):
        _, direction = parabolic_sar(large_df)
        assert set(direction.unique()).issubset({1, -1})

    def test_length(self, large_df):
        sar, direction = parabolic_sar(large_df)
        assert len(sar) == len(large_df) == len(direction)

    def test_sar_below_price_in_uptrend(self, trending_up_df):
        """In an uptrend the SAR line should be below close."""
        sar, direction = parabolic_sar(trending_up_df)
        # Check the last quarter where trend is firmly established
        tail = slice(-50, None)
        assert (sar.iloc[tail][direction.iloc[tail] == 1]
                <= trending_up_df["close"].iloc[tail][direction.iloc[tail] == 1]).all()

    def test_sar_above_price_in_downtrend(self, trending_down_df):
        sar, direction = parabolic_sar(trending_down_df)
        tail = slice(-50, None)
        assert (sar.iloc[tail][direction.iloc[tail] == -1]
                >= trending_down_df["close"].iloc[tail][direction.iloc[tail] == -1]).all()


# ============================================================================
# Strategy contract tests (all 15 new classes)
# ============================================================================

NEW_LONG_ONLY = [
    SupertrendStrategy, IchimokuStrategy, WilliamsRStrategy,
    DonchianBreakout, PSARStrategy,
]
NEW_LS = [
    SupertrendLS, IchimokuLS, WilliamsRLS,
    DonchianBreakoutLS, PSARLS,
]
NEW_REGIME = [
    SupertrendRegime, IchimokuRegime, WilliamsRRegime,
    DonchianBreakoutRegime, PSARRegime,
]


class TestNewStrategyContracts:
    @pytest.mark.parametrize("cls", NEW_LONG_ONLY)
    def test_long_only_returns_2tuple(self, cls, large_df):
        sigs = cls().generate_signals(large_df)
        assert len(sigs) == 2

    @pytest.mark.parametrize("cls", NEW_LS + NEW_REGIME)
    def test_ls_returns_4tuple(self, cls, large_df):
        sigs = cls().generate_signals(large_df)
        assert len(sigs) == 4

    @pytest.mark.parametrize("cls", NEW_LONG_ONLY + NEW_LS + NEW_REGIME)
    def test_signals_are_bool(self, cls, large_df):
        sigs = cls().generate_signals(large_df)
        for s in sigs:
            assert s.dtype == bool, f"{cls.__name__}: signal dtype {s.dtype} != bool"

    @pytest.mark.parametrize("cls", NEW_LONG_ONLY + NEW_LS + NEW_REGIME)
    def test_signals_match_df_length(self, cls, large_df):
        sigs = cls().generate_signals(large_df)
        for s in sigs:
            assert len(s) == len(large_df)

    @pytest.mark.parametrize("cls", NEW_LONG_ONLY)
    def test_long_only_runs_in_backtest(self, cls, large_df):
        """End-to-end smoke test: strategy + Backtest must not raise."""
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
        from backtest import Backtest
        bt = Backtest(initial=100_000)
        trades, equity, metrics = bt.run(large_df, cls())
        assert len(equity) == len(large_df) - 1
