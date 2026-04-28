import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI.

    Edge cases handled:
    - All-gains window (pure uptrend): avg_loss == 0  → RSI = 100
    - All-losses window (pure downtrend): avg_gain == 0 → RSI = 0
    - Both zero (flat/constant): RSI = 50 (neutral)
    """
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Build RSI bar-by-bar to handle the three edge cases cleanly
    rsi_vals = pd.Series(np.nan, index=series.index)

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    all_gain   = (avg_loss == 0) & (avg_gain > 0)
    all_loss   = (avg_gain == 0) & (avg_loss > 0)
    normal     = (avg_gain > 0) & (avg_loss > 0)

    rsi_vals[both_zero] = 50.0
    rsi_vals[all_gain]  = 100.0
    rsi_vals[all_loss]  = 0.0
    rs = avg_gain[normal] / avg_loss[normal]
    rsi_vals[normal] = 100 - (100 / (1 + rs))

    return rsi_vals


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    fast_ema = ema(series, int(fast))
    slow_ema = ema(series, int(slow))
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, int(signal))
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: int = 2,
) -> tuple:
    period = int(period)
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (int(std_dev) * std)
    lower = middle - (int(std_dev) * std)
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR        = EMA(TR, period)
    """
    period = int(period)
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (
        (typical_price * df["volume"]).rolling(window=20).sum()
        / df["volume"].rolling(window=20).sum()
    )


def trend_regime(
    series: pd.Series,
    ma_period: int = 200,
    slope_period: int = 20,
) -> pd.Series:
    """
    Classify each bar as bull (+1), bear (-1), or neutral (0).

    Method
    ------
    1. Compute a simple moving average of length *ma_period*.
    2. Measure the MA's slope over the last *slope_period* bars.
    3. slope > 0 → bull (+1); slope < 0 → bear (-1); warmup → neutral (0).
    """
    ma_period    = int(ma_period)
    slope_period = int(slope_period)
    ma = series.rolling(window=ma_period).mean()
    slope = (ma - ma.shift(slope_period)) / ma.shift(slope_period)
    regime = pd.Series(0, index=series.index, dtype=int)
    regime[slope > 0] = 1
    regime[slope < 0] = -1
    return regime
