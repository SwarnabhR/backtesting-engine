"""Unit tests for engine/indicators.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from indicators import ema, rsi, macd, bollinger_bands, atr, trend_regime


class TestEMA:
    def test_length(self, trending_up_df):
        result = ema(trending_up_df["close"], 10)
        assert len(result) == len(trending_up_df)

    def test_smoothing(self, trending_up_df):
        """EMA should be below close in an uptrend (lagging indicator)."""
        result = ema(trending_up_df["close"], 20)
        # after warmup the EMA should trail the rising price
        assert result.iloc[-1] < trending_up_df["close"].iloc[-1]

    def test_flat_equals_price(self, flat_df):
        """EMA of a flat series should equal that flat price."""
        result = ema(flat_df["close"], 10)
        np.testing.assert_allclose(result.iloc[20:], 100.0, rtol=1e-6)

    def test_no_nan_after_warmup(self, trending_up_df):
        period = 12
        result = ema(trending_up_df["close"], period)
        assert not result.iloc[period:].isna().any()


class TestRSI:
    def test_bounds(self, oscillating_df):
        result = rsi(oscillating_df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self, oscillating_df):
        assert len(rsi(oscillating_df["close"], 14)) == len(oscillating_df)

    def test_uptrend_high_rsi(self, trending_up_df):
        """A steep uptrend should produce RSI > 50 after warmup."""
        result = rsi(trending_up_df["close"], 14)
        assert result.dropna().iloc[-1] > 50

    def test_downtrend_low_rsi(self, trending_down_df):
        result = rsi(trending_down_df["close"], 14)
        assert result.dropna().iloc[-1] < 50


class TestMACD:
    def test_returns_three_series(self, trending_up_df):
        result = macd(trending_up_df["close"])
        assert len(result) == 3

    def test_lengths(self, trending_up_df):
        ml, sl, hist = macd(trending_up_df["close"])
        n = len(trending_up_df)
        assert len(ml) == len(sl) == len(hist) == n

    def test_histogram_definition(self, trending_up_df):
        ml, sl, hist = macd(trending_up_df["close"])
        pd.testing.assert_series_equal(hist, ml - sl, check_names=False)

    def test_uptrend_positive_macd(self, trending_up_df):
        ml, _, _ = macd(trending_up_df["close"])
        assert ml.dropna().iloc[-1] > 0


class TestBollingerBands:
    def test_returns_three_series(self, oscillating_df):
        assert len(bollinger_bands(oscillating_df["close"])) == 3

    def test_upper_above_lower(self, oscillating_df):
        upper, mid, lower = bollinger_bands(oscillating_df["close"])
        valid = upper.dropna()
        assert (valid > lower.dropna()).all()

    def test_mid_is_rolling_mean(self, oscillating_df):
        _, mid, _ = bollinger_bands(oscillating_df["close"], period=20)
        expected = oscillating_df["close"].rolling(20).mean()
        pd.testing.assert_series_equal(mid, expected, check_names=False)

    def test_flat_zero_bandwidth(self, flat_df):
        """A flat series has zero std → upper == lower == mid."""
        upper, mid, lower = bollinger_bands(flat_df["close"], period=10)
        # after warmup all three should be ~100
        np.testing.assert_allclose(upper.iloc[15:], 100.0, rtol=1e-5)
        np.testing.assert_allclose(lower.iloc[15:], 100.0, rtol=1e-5)


class TestATR:
    def test_length(self, trending_up_df):
        result = atr(trending_up_df)
        assert len(result) == len(trending_up_df)

    def test_non_negative(self, oscillating_df):
        result = atr(oscillating_df)
        assert (result.dropna() >= 0).all()

    def test_flat_small_atr(self, flat_df):
        """Flat prices have tiny ATR (only from the artificial noise)."""
        result = atr(flat_df)
        assert result.dropna().mean() < 5.0


class TestTrendRegime:
    def test_returns_series(self, large_df):
        result = trend_regime(large_df["close"])
        assert isinstance(result, pd.Series)

    def test_values_in_set(self, large_df):
        result = trend_regime(large_df["close"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_uptrend_bull(self, trending_up_df):
        """A clean uptrend should end in regime = +1."""
        result = trend_regime(trending_up_df["close"],
                              ma_period=50, slope_period=10)
        assert result.iloc[-1] == 1

    def test_downtrend_bear(self, trending_down_df):
        result = trend_regime(trending_down_df["close"],
                              ma_period=50, slope_period=10)
        assert result.iloc[-1] == -1
