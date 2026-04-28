"""Unit tests for engine/costs.py and cost integration in Backtest."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from costs import (
    NoCost, FixedCommission, PercentCommission,
    TickSlippage, PercentSlippage, CompositeCost,
    nse_equity_intraday, nse_equity_delivery,
)
from backtest import Backtest


# ── minimal buy-first / sell-last strategy ──────────────────────────────────

class _OneTrade:
    """Enter bar 1, exit bar -1.  Exactly one trade."""
    def generate_signals(self, df):
        e = pd.Series(False, index=df.index); e.iloc[1]  = True
        x = pd.Series(False, index=df.index); x.iloc[-1] = True
        return e, x


# ── unit tests: cost models in isolation ───────────────────────────────────

class TestNoCost:
    def test_always_zero(self):
        m = NoCost()
        assert m.cost(100, 10) == 0.0
        assert m.cost(0.001, 1_000_000) == 0.0


class TestFixedCommission:
    def test_flat_fee(self):
        m = FixedCommission(20.0)
        assert m.cost(100, 1)     == 20.0
        assert m.cost(100, 1_000) == 20.0   # shares don't matter

    def test_default_20(self):
        assert FixedCommission().cost(500, 5) == 20.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            FixedCommission(-1)


class TestPercentCommission:
    def test_known_value(self):
        # 0.1% of 100 * 10 shares = 0.1
        m = PercentCommission(pct=0.001)
        assert abs(m.cost(100, 10) - 1.0) < 1e-9

    def test_min_cost_floor(self):
        # tiny trade: pct gives 0.001, min_cost=1.0 kicks in
        m = PercentCommission(pct=0.001, min_cost=1.0)
        assert m.cost(0.1, 1) == 1.0

    def test_invalid_pct(self):
        with pytest.raises(ValueError):
            PercentCommission(pct=2.0)


class TestTickSlippage:
    def test_known_value(self):
        # 2 ticks * 0.05 tick_size * 100 shares = 10.0
        m = TickSlippage(ticks=2, tick_size=0.05)
        assert abs(m.cost(500, 100) - 10.0) < 1e-9

    def test_scales_with_shares(self):
        m = TickSlippage()
        assert m.cost(100, 10) == 2 * m.cost(100, 5)

    def test_invalid_ticks(self):
        with pytest.raises(ValueError):
            TickSlippage(ticks=-1)

    def test_invalid_tick_size(self):
        with pytest.raises(ValueError):
            TickSlippage(tick_size=0)


class TestPercentSlippage:
    def test_known_value(self):
        # 0.02% of 100 * 5 shares = 0.10
        m = PercentSlippage(pct=0.0002)
        assert abs(m.cost(100, 5) - 0.10) < 1e-9

    def test_invalid_pct(self):
        with pytest.raises(ValueError):
            PercentSlippage(pct=1.5)


class TestCompositeCost:
    def test_sums_components(self):
        # fixed 20 + 0.1% of 100*10 = 20 + 1 = 21
        m = CompositeCost(FixedCommission(20), PercentCommission(0.001))
        assert abs(m.cost(100, 10) - 21.0) < 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            CompositeCost()

    def test_three_components(self):
        m = CompositeCost(
            FixedCommission(10),
            PercentCommission(0.001),
            TickSlippage(1, 0.05),
        )
        # 10 + (100*10*0.001) + (1*0.05*10) = 10 + 1 + 0.5 = 11.5
        assert abs(m.cost(100, 10) - 11.5) < 1e-9


class TestPresets:
    def test_nse_intraday_nonzero(self):
        m = nse_equity_intraday()
        assert m.cost(100, 10) > 0

    def test_nse_delivery_nonzero(self):
        m = nse_equity_delivery()
        assert m.cost(100, 10) > 0


# ── integration: cost model wired into Backtest ───────────────────────────

class TestBacktestIntegration:
    def test_cost_reduces_pnl(self, trending_up_df):
        """Net P&L with costs < gross P&L without costs."""
        bt_free = Backtest(initial=10_000)
        bt_cost = Backtest(initial=10_000,
                           cost_model=FixedCommission(50.0))
        t_free, _, _ = bt_free.run(trending_up_df, _OneTrade())
        t_cost, _, _ = bt_cost.run(trending_up_df, _OneTrade())
        assert t_cost.iloc[0]["pnl"] < t_free.iloc[0]["pnl"]

    def test_cost_column_in_trades(self, trending_up_df):
        """trades_df must have a 'cost' column when cost_model is set."""
        bt = Backtest(initial=10_000, cost_model=FixedCommission(20.0))
        trades, _, _ = bt.run(trending_up_df, _OneTrade())
        assert "cost" in trades.columns
        assert trades.iloc[0]["cost"] == 40.0   # entry + exit

    def test_gross_pnl_column(self, trending_up_df):
        bt = Backtest(initial=10_000, cost_model=PercentSlippage(0.001))
        trades, _, _ = bt.run(trending_up_df, _OneTrade())
        assert "gross_pnl" in trades.columns
        row = trades.iloc[0]
        assert abs(row["gross_pnl"] - row["pnl"] - row["cost"]) < 1e-6

    def test_no_cost_backward_compat(self, trending_up_df):
        """Default Backtest() (no cost_model kwarg) behaves as NoCost."""
        bt_old = Backtest(initial=10_000)
        bt_new = Backtest(initial=10_000, cost_model=NoCost())
        t_old, eq_old, _ = bt_old.run(trending_up_df, _OneTrade())
        t_new, eq_new, _ = bt_new.run(trending_up_df, _OneTrade())
        np.testing.assert_allclose(
            t_old.iloc[0]["pnl"], t_new.iloc[0]["pnl"]
        )

    def test_high_cost_wipes_profit(self, flat_df):
        """Flat market + huge fixed cost = negative net P&L."""
        bt = Backtest(initial=10_000, cost_model=FixedCommission(1_000_000))
        trades, _, _ = bt.run(flat_df, _OneTrade())
        if not trades.empty:
            assert trades.iloc[0]["pnl"] < 0

    def test_equity_lower_with_costs(self, trending_up_df):
        """Final equity must be lower when costs are applied."""
        bt_free = Backtest(initial=10_000)
        bt_cost = Backtest(initial=10_000, cost_model=nse_equity_intraday())
        _, eq_free, _ = bt_free.run(trending_up_df, _OneTrade())
        _, eq_cost, _ = bt_cost.run(trending_up_df, _OneTrade())
        assert eq_cost.iloc[-1] <= eq_free.iloc[-1]
