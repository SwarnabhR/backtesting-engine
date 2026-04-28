"""Unit tests for engine/optimizer.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from optimizer import GridOptimizer, OptimizationResult
from strategy import EMACrossover


class TestGridOptimizer:
    def test_returns_dataframe(self, large_df):
        opt = GridOptimizer(
            strategy_class=EMACrossover,
            param_grid={"fast": [5, 10], "slow": [20, 30]},
            min_trades=1,
            constraint=lambda p: p["fast"] < p["slow"],
        )
        result = opt.run(large_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_constraint_respected(self, large_df):
        opt = GridOptimizer(
            strategy_class=EMACrossover,
            param_grid={"fast": [5, 10, 20], "slow": [10, 20, 30]},
            min_trades=1,
            constraint=lambda p: p["fast"] < p["slow"],
        )
        df_result = opt.run(large_df)
        for _, row in df_result.iterrows():
            assert row["fast"] < row["slow"]

    def test_sorted_by_sharpe(self, large_df):
        opt = GridOptimizer(
            strategy_class=EMACrossover,
            param_grid={"fast": [5, 10], "slow": [20, 30]},
            min_trades=1,
            constraint=lambda p: p["fast"] < p["slow"],
        )
        df_result = opt.run(large_df, sort_by="sharpe_ratio")
        sharpes = df_result["sharpe_ratio"].tolist()
        assert sharpes == sorted(sharpes, reverse=True)

    def test_best_returns_result(self, large_df):
        opt = GridOptimizer(
            strategy_class=EMACrossover,
            param_grid={"fast": [5, 10], "slow": [20, 40]},
            min_trades=1,
            constraint=lambda p: p["fast"] < p["slow"],
        )
        best = opt.best(large_df)
        assert isinstance(best, OptimizationResult)
        assert "fast" in best.params
        assert "slow" in best.params

    def test_min_trades_filter(self, large_df):
        """With min_trades=9999, result should be empty."""
        opt = GridOptimizer(
            strategy_class=EMACrossover,
            param_grid={"fast": [5], "slow": [20]},
            min_trades=9_999,
        )
        result = opt.run(large_df)
        assert result.empty
