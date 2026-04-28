"""Unit tests for engine/sizer.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from sizer import FixedSizer, PercentEquitySizer, ATRSizer, KellySizer


def _bar(close=100.0, atr=5.0) -> pd.Series:
    return pd.Series({"open": close, "high": close+1, "low": close-1,
                      "close": close, "volume": 1000.0, "atr": atr})


class TestFixedSizer:
    def test_always_n(self):
        s = FixedSizer(shares=5)
        assert s.size(10_000, _bar()) == 5

    def test_default_one(self):
        assert FixedSizer().size(10_000, _bar()) == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            FixedSizer(shares=0)


class TestPercentEquitySizer:
    def test_formula(self):
        # equity=10_000, pct=0.10, close=100 → floor(1000/100) = 10
        s = PercentEquitySizer(pct=0.10, min_shares=1)
        assert s.size(10_000, _bar(close=100)) == 10

    def test_min_shares_floor(self):
        # allocation rounds to 0 → min_shares=1 kicks in
        s = PercentEquitySizer(pct=0.01, min_shares=1)
        assert s.size(100, _bar(close=1_000)) == 1

    def test_invalid_pct(self):
        with pytest.raises(ValueError):
            PercentEquitySizer(pct=0)


class TestATRSizer:
    def test_known_result(self):
        # equity=100_000, risk_pct=0.01, ATR=150, atr_mult=2.0
        # stop = 300, budget = 1000 → floor(1000/300) = 3
        s = ATRSizer(risk_pct=0.01, atr_mult=2.0)
        assert s.size(100_000, _bar(atr=150)) == 3

    def test_zero_atr_fallback(self):
        s = ATRSizer(risk_pct=0.01, atr_mult=2.0, min_shares=1)
        assert s.size(100_000, _bar(atr=0)) == 1

    def test_invalid_risk_pct(self):
        with pytest.raises(ValueError):
            ATRSizer(risk_pct=0)

    def test_scales_with_equity(self):
        s = ATRSizer(risk_pct=0.01, atr_mult=2.0)
        low  = s.size(50_000,  _bar(atr=100))
        high = s.size(200_000, _bar(atr=100))
        assert high > low


class TestKellySizer:
    def test_known_fraction(self):
        # b = 0.03/0.015 = 2, p=0.6, q=0.4
        # raw_f = (2*0.6 - 0.4)/2 = 0.4, half-Kelly = 0.2
        # shares = floor(10_000 * 0.2 / 100) = 20
        s = KellySizer(win_rate=0.6, avg_win=0.03, avg_loss=0.015, fraction=0.5)
        assert s.size(10_000, _bar(close=100)) == 20

    def test_negative_kelly_min_shares(self):
        """Edge case: losing strategy → Kelly f* < 0 → clamp to min_shares."""
        s = KellySizer(win_rate=0.2, avg_win=0.01, avg_loss=0.05, fraction=0.5)
        assert s.size(10_000, _bar()) == 1

    def test_from_trades(self):
        trades = pd.DataFrame({
            "pnl":     [100, -50, 200, -30, 150],
            "pnl_pct": [0.05, -0.025, 0.10, -0.015, 0.075],
        })
        s = KellySizer.from_trades(trades, fraction=0.5)
        assert s.size(10_000, _bar()) >= 1

    def test_invalid_win_rate(self):
        with pytest.raises(ValueError):
            KellySizer(win_rate=1.0, avg_win=0.02, avg_loss=0.01)
