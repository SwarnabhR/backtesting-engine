import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
    2. Measure the MA’s slope over the last *slope_period* bars as a
       percentage:  slope = (MA_now - MA_n_bars_ago) / MA_n_bars_ago
    3. Threshold:
         slope >  0  → bull  (+1)  — MA is rising
         slope <  0  → bear  (-1)  — MA is falling
         NaN (warmup period) → neutral (0)

    Using the *slope* rather than a raw price-vs-MA comparison avoids
    whipsaw during sideways markets where price oscillates around a flat MA.

    Parameters
    ----------
    series : pd.Series
        Close prices.
    ma_period : int
        Lookback for the trend-defining moving average (default 200).
    slope_period : int
        Number of bars over which the MA slope is measured (default 20).

    Returns
    -------
    pd.Series of int  {-1, 0, +1}  aligned to *series* index.
    """
    # Explicit int() casts guard against float values arriving from the
    # optimizer's Cartesian product (e.g. 10.0 instead of 10), which
    # would cause pd.Series.shift() to raise TypeError.
    ma_period   = int(ma_period)
    slope_period = int(slope_period)

    ma = series.rolling(window=ma_period).mean()
    slope = (ma - ma.shift(slope_period)) / ma.shift(slope_period)

    regime = pd.Series(0, index=series.index, dtype=int)
    regime[slope > 0] = 1
    regime[slope < 0] = -1
    return regime
