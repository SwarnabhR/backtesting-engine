"""Unit tests for engine/strategy.py — signal contract and basic behaviour."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from strategy import (
    EMACrossover, EMACrossoverLS, EMACrossoverRegime,
    RSIMeanReversion, RSIMeanReversionLS, RSIMeanReversionRegime,
    BollingerBreakout, BollingerBreakoutLS, BollingerBreakoutRegime,
    MACDCrossover, MACDCrossoverLS, MACDCrossoverRegime,
)

LONG_ONLY_STRATEGIES  = [
    EMACrossover, RSIMeanReversion, BollingerBreakout, MACDCrossover
]
LONG_SHORT_STRATEGIES = [
    EMACrossoverLS, RSIMeanReversionLS, BollingerBreakoutLS, MACDCrossoverLS
]
REGIME_STRATEGIES     = [
    EMACrossoverRegime, RSIMeanReversionRegime,
    BollingerBreakoutRegime, MACDCrossoverRegime
]


class TestSignalContract:
    def test_long_only_returns_2tuple(self, large_df):
        for cls in LONG_ONLY_STRATEGIES:
            sigs = cls().generate_signals(large_df)
            assert len(sigs) == 2, f"{cls.__name__} must return 2-tuple"

    def test_long_short_returns_4tuple(self, large_df):
        for cls in LONG_SHORT_STRATEGIES:
            sigs = cls().generate_signals(large_df)
            assert len(sigs) == 4, f"{cls.__name__} must return 4-tuple"

    def test_regime_returns_4tuple(self, large_df):
        for cls in REGIME_STRATEGIES:
            sigs = cls().generate_signals(large_df)
            assert len(sigs) == 4, f"{cls.__name__} must return 4-tuple"

    def test_signal_index_matches_df(self, large_df):
        for cls in LONG_ONLY_STRATEGIES + LONG_SHORT_STRATEGIES + REGIME_STRATEGIES:
            sigs = cls().generate_signals(large_df)
            for s in sigs:
                assert len(s) == len(large_df), (
                    f"{cls.__name__} signal length mismatch"
                )

    def test_signals_are_boolean(self, large_df):
        for cls in LONG_ONLY_STRATEGIES:
            entries, exits = cls().generate_signals(large_df)
            assert entries.dtype == bool, f"{cls.__name__} entries not bool"
            assert exits.dtype == bool,   f"{cls.__name__} exits not bool"


class TestEMACrossover:
    def test_fires_on_step_up(self, step_up_df):
        """Golden cross in step_up_df must produce at least one entry."""
        entries, _ = EMACrossover(fast=10, slow=30).generate_signals(step_up_df)
        assert entries.any()

    def test_no_trades_flat(self, flat_df):
        """Flat price → no crossovers → no entries."""
        entries, _ = EMACrossover(fast=5, slow=20).generate_signals(flat_df)
        # flat price may still produce 1 spurious entry at init; allow ≤1
        assert entries.sum() <= 1


class TestRSIMeanReversion:
    def test_fires_on_oscillating(self, oscillating_df):
        entries, exits = RSIMeanReversion().generate_signals(oscillating_df)
        assert entries.any() or exits.any()


class TestBollingerBreakout:
    def test_no_crash_on_all_fixtures(
        self, flat_df, trending_up_df, oscillating_df
    ):
        for df in [flat_df, trending_up_df, oscillating_df]:
            BollingerBreakout().generate_signals(df)  # must not raise


class TestMACDCrossover:
    def test_fires_on_large(self, large_df):
        entries, exits = MACDCrossover().generate_signals(large_df)
        assert entries.any()
