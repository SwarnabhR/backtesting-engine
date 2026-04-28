import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI.

    Edge cases:
    - All-gains (pure uptrend): avg_loss == 0  -> RSI = 100
    - All-losses (pure downtrend): avg_gain == 0 -> RSI = 0
    - Both zero (flat): RSI = 50
    """
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rsi_vals = pd.Series(np.nan, index=series.index)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    all_gain  = (avg_loss == 0) & (avg_gain > 0)
    all_loss  = (avg_gain == 0) & (avg_loss > 0)
    normal    = (avg_gain > 0)  & (avg_loss > 0)

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
    fast_ema   = ema(series, int(fast))
    slow_ema   = ema(series, int(slow))
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, int(signal))
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: int = 2,
) -> tuple:
    period = int(period)
    middle = series.rolling(window=period).mean()
    std    = series.rolling(window=period).std()
    upper  = middle + (int(std_dev) * std)
    lower  = middle - (int(std_dev) * std)
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range  (EMA smoothed).
    Requires columns: high, low, close.
    """
    period     = int(period)
    high       = df["high"]
    low        = df["low"]
    close      = df["close"]
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
    Classify each bar as bull (+1), bear (-1), or neutral (0) using
    a simple moving average slope.
    """
    ma_period    = int(ma_period)
    slope_period = int(slope_period)
    ma    = series.rolling(window=ma_period).mean()
    slope = (ma - ma.shift(slope_period)) / ma.shift(slope_period)
    regime = pd.Series(0, index=series.index, dtype=int)
    regime[slope > 0] =  1
    regime[slope < 0] = -1
    return regime


# ============================================================================
# NEW INDICATORS
# ============================================================================

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.

    Returns
    -------
    supertrend : pd.Series  — the Supertrend line value at each bar
    direction  : pd.Series  — +1 = uptrend (bullish), -1 = downtrend (bearish)

    Algorithm
    ---------
    1. Basic upper/lower bands around the HL midpoint +/- multiplier*ATR.
    2. Final bands are tightened bar-by-bar (never widen when price stays
       on the same side).
    3. Direction flips when price closes on the opposite side of the band.
    """
    atr_vals = atr(df, period)
    hl_avg   = (df["high"] + df["low"]) / 2
    close    = df["close"]

    basic_upper = hl_avg + multiplier * atr_vals
    basic_lower = hl_avg - multiplier * atr_vals

    n = len(df)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction   = pd.Series(1, index=df.index, dtype=int)
    st          = pd.Series(np.nan, index=df.index)

    for i in range(1, n):
        # tighten upper band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # tighten lower band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # determine direction
        if direction.iloc[i - 1] == -1 and close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i - 1] == 1 and close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return st, direction


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> dict[str, pd.Series]:
    """
    Ichimoku Kinko Hyo.

    Returns a dict with keys:
      tenkan_sen    — Conversion line  (9-bar midpoint)
      kijun_sen     — Base line        (26-bar midpoint)
      senkou_a      — Leading span A   (avg of tenkan + kijun, shifted +26)
      senkou_b      — Leading span B   (52-bar midpoint, shifted +26)
      chikou        — Lagging span     (close shifted -26)
    """
    def _midpoint(s: pd.Series, p: int) -> pd.Series:
        return (s.rolling(p).max() + s.rolling(p).min()) / 2

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tenkan_sen = _midpoint(high, tenkan) - (_midpoint(high, tenkan) - _midpoint(low, tenkan))
    # simpler direct formula:
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    senkou_a   = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_s = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou     = close.shift(-kijun)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen":  kijun_sen,
        "senkou_a":   senkou_a,
        "senkou_b":   senkou_b_s,
        "chikou":     chikou,
    }


def williams_r(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R oscillator.  Range: [-100, 0].
    -80 to -100 = oversold; 0 to -20 = overbought.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    """
    period     = int(period)
    high_max   = df["high"].rolling(period).max()
    low_min    = df["low"].rolling(period).min()
    denom      = high_max - low_min
    wr         = (high_max - df["close"]) / denom.replace(0, np.nan) * -100
    return wr


def donchian_channels(
    df: pd.DataFrame,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels.

    Returns
    -------
    upper  : rolling N-bar high
    middle : midpoint of upper and lower
    lower  : rolling N-bar low
    """
    period = int(period)
    upper  = df["high"].rolling(period).max()
    lower  = df["low"].rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def parabolic_sar(
    df: pd.DataFrame,
    initial_af: float = 0.02,
    max_af: float = 0.20,
    step_af: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """
    Parabolic SAR (J. Welles Wilder).

    Returns
    -------
    sar       : pd.Series  — SAR value at each bar
    direction : pd.Series  — +1 = uptrend, -1 = downtrend
    """
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    sar    = np.zeros(n)
    ep     = np.zeros(n)        # extreme point
    af_arr = np.zeros(n)        # acceleration factor
    trend  = np.ones(n, dtype=int)  # +1 / -1

    # Initialise with bar 0
    sar[0]    = low[0]
    ep[0]     = high[0]
    af_arr[0] = initial_af
    trend[0]  = 1

    for i in range(1, n):
        prev_sar   = sar[i - 1]
        prev_ep    = ep[i - 1]
        prev_af    = af_arr[i - 1]
        prev_trend = trend[i - 1]

        # Project SAR
        new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

        if prev_trend == 1:          # uptrend
            new_sar = min(new_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < new_sar:     # reversal to downtrend
                trend[i]  = -1
                sar[i]    = prev_ep
                ep[i]     = low[i]
                af_arr[i] = initial_af
            else:
                trend[i] = 1
                sar[i]   = new_sar
                if high[i] > prev_ep:
                    ep[i]     = high[i]
                    af_arr[i] = min(prev_af + step_af, max_af)
                else:
                    ep[i]     = prev_ep
                    af_arr[i] = prev_af
        else:                        # downtrend
            new_sar = max(new_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > new_sar:    # reversal to uptrend
                trend[i]  = 1
                sar[i]    = prev_ep
                ep[i]     = high[i]
                af_arr[i] = initial_af
            else:
                trend[i] = -1
                sar[i]   = new_sar
                if low[i] < prev_ep:
                    ep[i]     = low[i]
                    af_arr[i] = min(prev_af + step_af, max_af)
                else:
                    ep[i]     = prev_ep
                    af_arr[i] = prev_af

    return (
        pd.Series(sar,   index=df.index),
        pd.Series(trend, index=df.index, dtype=int),
    )
