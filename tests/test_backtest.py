"""Unit tests for engine/backtest.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from backtest import Backtest
from sizer import FixedSizer, PercentEquitySizer, ATRSizer
from indicators import atr as compute_atr


# ── minimal dummy strategies ───────────────────────────────────────────────────

class _BuyFirstSellLast:
    """Enter on bar 1, exit on last bar — exactly 1 trade."""
    def generate_signals(self, df):
        entries = pd.Series(False, index=df.index)
        exits   = pd.Series(False, index=df.index)
        entries.iloc[1] = True
        exits.iloc[-1]  = True
        return entries, exits


class _NeverTrade:
    """All signals are False — zero trades."""
    def generate_signals(self, df):
        f = pd.Series(False, index=df.index)
        return f.copy(), f.copy()


class _AlwaysInMarket:
    """Buy every bar, sell every bar (real engine allows only 1 position)."""
    def generate_signals(self, df):
        t = pd.Series(True, index=df.index)
        return t.copy(), t.copy()


class _LongShortAlternate:
    """4-tuple: buy at bar 2, exit long at bar 5, short at bar 5, cover at last."""
    def generate_signals(self, df):
        n = len(df)
        le = pd.Series(False, index=df.index); le.iloc[2]  = True
        lx = pd.Series(False, index=df.index); lx.iloc[5]  = True
        se = pd.Series(False, index=df.index); se.iloc[5]  = True
        sx = pd.Series(False, index=df.index); sx.iloc[-1] = True
        return le, lx, se, sx


# ── tests ───────────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_equity_length(self, trending_up_df):
        bt = Backtest(initial=10_000)
        _, equity, _ = bt.run(trending_up_df, _BuyFirstSellLast())
        # equity has one entry per bar starting from index 1
        assert len(equity) == len(trending_up_df) - 1

    def test_trades_columns(self, trending_up_df):
        bt = Backtest(initial=10_000)
        trades, _, _ = bt.run(trending_up_df, _BuyFirstSellLast())
        expected = {"entry_date", "exit_date", "entry_price",
                    "exit_price", "direction", "shares", "pnl", "pnl_pct"}
        assert expected.issubset(set(trades.columns))

    def test_zero_trades_empty_df(self, trending_up_df):
        bt = Backtest(initial=10_000)
        trades, _, _ = bt.run(trending_up_df, _NeverTrade())
        assert trades.empty


class TestPnLAccounting:
    def test_long_profit_positive(self, trending_up_df):
        """Buy at bar 1 in an uptrend, sell last → pnl > 0."""
        bt = Backtest(initial=10_000)
        trades, equity, _ = bt.run(trending_up_df, _BuyFirstSellLast())
        assert trades.iloc[0]["pnl"] > 0
        assert equity.iloc[-1] > 10_000

    def test_long_loss_downtrend(self, trending_down_df):
        bt = Backtest(initial=10_000)
        trades, equity, _ = bt.run(trending_down_df, _BuyFirstSellLast())
        assert trades.iloc[0]["pnl"] < 0
        assert equity.iloc[-1] < 10_000

    def test_equity_flat_no_trades(self, flat_df):
        bt = Backtest(initial=10_000)
        _, equity, _ = bt.run(flat_df, _NeverTrade())
        # equity should stay constant at initial capital
        np.testing.assert_allclose(equity.values, 10_000.0)

    def test_pnl_pct_formula_long(self, trending_up_df):
        bt = Backtest(initial=10_000)
        trades, _, _ = bt.run(trending_up_df, _BuyFirstSellLast())
        row = trades.iloc[0]
        expected_pct = (row["exit_price"] - row["entry_price"]) / row["entry_price"]
        assert abs(row["pnl_pct"] - expected_pct) < 1e-9

    def test_long_short_produces_two_trades(self, large_df):
        bt = Backtest(initial=10_000)
        trades, _, _ = bt.run(large_df, _LongShortAlternate())
        assert len(trades) == 2
        assert trades.iloc[0]["direction"] == "long"
        assert trades.iloc[1]["direction"] == "short"

    def test_short_profit_downtrend(self, trending_down_df):
        """Short sell at bar 2, cover at bar 5 in downtrend → profit."""
        class _ShortOnly:
            def generate_signals(self, df):
                se = pd.Series(False, index=df.index); se.iloc[2] = True
                sx = pd.Series(False, index=df.index); sx.iloc[5] = True
                le = pd.Series(False, index=df.index)
                lx = pd.Series(False, index=df.index)
                return le, lx, se, sx
        bt = Backtest(initial=10_000)
        trades, _, _ = bt.run(trending_down_df, _ShortOnly())
        assert trades.iloc[0]["pnl"] > 0


class TestPositionSizing:
    def test_fixed_sizer_shares(self, trending_up_df):
        bt = Backtest(initial=10_000, sizer=FixedSizer(shares=3))
        trades, _, _ = bt.run(trending_up_df, _BuyFirstSellLast())
        assert trades.iloc[0]["shares"] == 3

    def test_atr_sizer_nonzero(self, trending_up_df):
        df = trending_up_df.copy()
        df["atr"] = compute_atr(df, 14)
        bt = Backtest(initial=100_000, sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0))
        trades, _, _ = bt.run(df, _BuyFirstSellLast())
        assert trades.iloc[0]["shares"] >= 1

    def test_percent_equity_scales_with_price(self, trending_up_df):
        """Higher equity -> more shares for same pct."""
        sizer = PercentEquitySizer(pct=0.50, min_shares=1)
        bt1 = Backtest(initial=10_000,  sizer=sizer)
        bt2 = Backtest(initial=100_000, sizer=sizer)
        t1, _, _ = bt1.run(trending_up_df, _BuyFirstSellLast())
        t2, _, _ = bt2.run(trending_up_df, _BuyFirstSellLast())
        assert t2.iloc[0]["shares"] >= t1.iloc[0]["shares"]

    def test_backward_compat_position_size(self, trending_up_df):
        """position_size=N kwarg should behave identically to FixedSizer(N)."""
        bt_old = Backtest(initial=10_000, position_size=2)
        bt_new = Backtest(initial=10_000, sizer=FixedSizer(2))
        t_old, _, _ = bt_old.run(trending_up_df, _BuyFirstSellLast())
        t_new, _, _ = bt_new.run(trending_up_df, _BuyFirstSellLast())
        assert t_old.iloc[0]["shares"] == t_new.iloc[0]["shares"]
        np.testing.assert_allclose(t_old.iloc[0]["pnl"], t_new.iloc[0]["pnl"])


class TestEdgeCases:
    def test_short_data_no_crash(self, short_df):
        bt = Backtest(initial=10_000)
        trades, equity, metrics = bt.run(short_df, _BuyFirstSellLast())
        assert isinstance(metrics.sharpe_ratio, float)

    def test_invalid_signal_tuple_raises(self, flat_df):
        class _BadStrategy:
            def generate_signals(self, df):
                return (pd.Series(False, index=df.index),)  # 1-tuple
        bt = Backtest(initial=10_000)
        with pytest.raises(ValueError):
            bt.run(flat_df, _BadStrategy())
