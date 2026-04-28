"""Shared synthetic OHLCV fixtures used across all test modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    close: list[float] | np.ndarray,
    seed: int = 0,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price array."""
    rng = np.random.default_rng(seed)
    close = np.asarray(close, dtype=float)
    n = len(close)
    noise = rng.uniform(0.001, 0.005, n) * close
    high   = close + noise
    low    = close - noise
    open_  = np.roll(close, 1);  open_[0] = close[0]
    volume = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_df() -> pd.DataFrame:
    """100 bars of perfectly flat price = 100.  No signal should fire."""
    return _make_df([100.0] * 100)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """300 bars of a clean uptrend: 100 → 400 linearly."""
    return _make_df(np.linspace(100, 400, 300))


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """300 bars of a clean downtrend: 400 → 100 linearly."""
    return _make_df(np.linspace(400, 100, 300))


@pytest.fixture
def step_up_df() -> pd.DataFrame:
    """
    Engineered golden cross: 150 bars flat at 100,
    then 150 bars at 200.  Forces EMA(fast) to cross EMA(slow) upward
    at the step.
    """
    prices = [100.0] * 150 + [200.0] * 150
    return _make_df(prices)


@pytest.fixture
def step_down_df() -> pd.DataFrame:
    """Engineered death cross: 150 bars at 200, then 150 bars at 100."""
    prices = [200.0] * 150 + [100.0] * 150
    return _make_df(prices)


@pytest.fixture
def oscillating_df() -> pd.DataFrame:
    """
    200 bars of a sine wave oscillating between 80 and 120.
    RSI and Bollinger strategies should fire on this data.
    """
    t = np.linspace(0, 4 * np.pi, 200)
    close = 100 + 20 * np.sin(t)
    return _make_df(close)


@pytest.fixture
def short_df() -> pd.DataFrame:
    """Only 10 bars — too short for any strategy to produce trades."""
    return _make_df(np.linspace(100, 110, 10))


@pytest.fixture
def large_df() -> pd.DataFrame:
    """1500 bars of GBM-like data for performance / WFV tests."""
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0003, 0.012, 1500)
    close = 100 * np.exp(np.cumsum(log_returns))
    return _make_df(close)
