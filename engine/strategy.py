from abc import ABC, abstractmethod
import pandas as pd
from indicators import (
    ema, rsi, bollinger_bands, macd, trend_regime,
    supertrend, ichimoku, williams_r, donchian_channels, parabolic_sar,
)


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Return either:
          - 2-tuple (entries, exits)                          → long-only
          - 4-tuple (long_entries, long_exits,
                     short_entries, short_exits)              → long/short
        """


# ─────────────────────────────────────────────────────────────────────────────
# EMA Crossover
# ─────────────────────────────────────────────────────────────────────────────

class EMACrossover(Strategy):
    """Long-only: buy on golden cross, sell on death cross."""
    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast; self.slow = slow

    def generate_signals(self, df):
        close = df["close"]
        fe = ema(close, self.fast); se = ema(close, self.slow)
        entries = (fe > se) & (fe.shift(1) <= se.shift(1))
        exits   = (fe < se) & (fe.shift(1) >= se.shift(1))
        return entries, exits


class EMACrossoverLS(Strategy):
    """Long/short EMA crossover (always in market)."""
    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast; self.slow = slow

    def generate_signals(self, df):
        close = df["close"]
        fe = ema(close, self.fast); se = ema(close, self.slow)
        golden = (fe > se) & (fe.shift(1) <= se.shift(1))
        death  = (fe < se) & (fe.shift(1) >= se.shift(1))
        return golden, death, death, golden


class EMACrossoverRegime(Strategy):
    """Regime-filtered EMA crossover."""
    def __init__(self, fast=12, slow=26, regime_ma=200, regime_slope=20):
        self.fast = fast; self.slow = slow
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df):
        close = df["close"]
        fe = ema(close, self.fast); se = ema(close, self.slow)
        golden = (fe > se) & (fe.shift(1) <= se.shift(1))
        death  = (fe < se) & (fe.shift(1) >= se.shift(1))
        regime = trend_regime(close, self.regime_ma, self.regime_slope)
        bull = regime == 1; bear = regime == -1
        return golden & bull, death, death & bear, golden


# ─────────────────────────────────────────────────────────────────────────────
# RSI Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

class RSIMeanReversion(Strategy):
    """Long-only RSI mean reversion."""
    def __init__(self, oversold=30, overbought=70, period=14):
        self.oversold = oversold; self.overbought = overbought; self.period = period

    def generate_signals(self, df):
        r = rsi(df["close"], self.period)
        return (
            (r > self.oversold)   & (r.shift(1) <= self.oversold),
            (r > self.overbought) & (r.shift(1) <= self.overbought),
        )


class RSIMeanReversionLS(Strategy):
    """Long/short RSI."""
    def __init__(self, oversold=30, overbought=70, period=14):
        self.oversold = oversold; self.overbought = overbought; self.period = period

    def generate_signals(self, df):
        r = rsi(df["close"], self.period)
        le = (r > self.oversold)   & (r.shift(1) <= self.oversold)
        lx = (r > self.overbought) & (r.shift(1) <= self.overbought)
        se = (r < self.overbought) & (r.shift(1) >= self.overbought)
        sx = (r < self.oversold)   & (r.shift(1) >= self.oversold)
        return le, lx, se, sx


class RSIMeanReversionRegime(Strategy):
    """Regime-filtered RSI."""
    def __init__(self, oversold=30, overbought=70, period=14, regime_ma=200, regime_slope=20):
        self.oversold = oversold; self.overbought = overbought; self.period = period
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df):
        r  = rsi(df["close"], self.period)
        reg = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        le = (r > self.oversold)   & (r.shift(1) <= self.oversold)  & bull
        lx = (r > self.overbought) & (r.shift(1) <= self.overbought)
        se = (r < self.overbought) & (r.shift(1) >= self.overbought) & bear
        sx = (r < self.oversold)   & (r.shift(1) >= self.oversold)
        return le, lx, se, sx


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

class BollingerBreakout(Strategy):
    """Long-only Bollinger breakout."""
    def __init__(self, period=20, std_dev=2):
        self.period = period; self.std_dev = std_dev

    def generate_signals(self, df):
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std_dev)
        c = df["close"]
        return (
            (c > upper)  & (c.shift(1) <= upper.shift(1)),
            (c < middle) & (c.shift(1) >= middle.shift(1)),
        )


class BollingerBreakoutLS(Strategy):
    """Long/short Bollinger."""
    def __init__(self, period=20, std_dev=2):
        self.period = period; self.std_dev = std_dev

    def generate_signals(self, df):
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std_dev)
        c = df["close"]
        le = (c > upper)  & (c.shift(1) <= upper.shift(1))
        lx = (c < middle) & (c.shift(1) >= middle.shift(1))
        se = (c < lower)  & (c.shift(1) >= lower.shift(1))
        sx = (c > middle) & (c.shift(1) <= middle.shift(1))
        return le, lx, se, sx


class BollingerBreakoutRegime(Strategy):
    """Regime-filtered Bollinger."""
    def __init__(self, period=20, std_dev=2, regime_ma=200, regime_slope=20):
        self.period = period; self.std_dev = std_dev
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df):
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std_dev)
        c = df["close"]
        reg = trend_regime(c, self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        return (
            (c > upper)  & (c.shift(1) <= upper.shift(1))  & bull,
            (c < middle) & (c.shift(1) >= middle.shift(1)),
            (c < lower)  & (c.shift(1) >= lower.shift(1))  & bear,
            (c > middle) & (c.shift(1) <= middle.shift(1)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MACD Crossover
# ─────────────────────────────────────────────────────────────────────────────

class MACDCrossover(Strategy):
    """Long-only MACD signal-line crossover."""
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast; self.slow = slow; self.signal = signal

    def generate_signals(self, df):
        ml, sl, _ = macd(df["close"], self.fast, self.slow, self.signal)
        return (
            (ml > sl) & (ml.shift(1) <= sl.shift(1)),
            (ml < sl) & (ml.shift(1) >= sl.shift(1)),
        )


class MACDCrossoverLS(Strategy):
    """Long/short MACD."""
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast; self.slow = slow; self.signal = signal

    def generate_signals(self, df):
        ml, sl, _ = macd(df["close"], self.fast, self.slow, self.signal)
        le = (ml > sl) & (ml.shift(1) <= sl.shift(1))
        lx = (ml < sl) & (ml.shift(1) >= sl.shift(1))
        return le, lx, lx, le


class MACDCrossoverRegime(Strategy):
    """Regime-filtered MACD crossover."""
    def __init__(self, fast=12, slow=26, signal=9, regime_ma=200, regime_slope=20):
        self.fast = fast; self.slow = slow; self.signal = signal
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df):
        ml, sl, _ = macd(df["close"], self.fast, self.slow, self.signal)
        reg = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        bull_x = (ml > sl) & (ml.shift(1) <= sl.shift(1))
        bear_x = (ml < sl) & (ml.shift(1) >= sl.shift(1))
        return bull_x & bull, bear_x, bear_x & bear, bull_x


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────────────────────────────────────

class SupertrendStrategy(Strategy):
    """
    Long-only Supertrend.
    Enter when direction flips to +1 (uptrend), exit when it flips to -1.
    """
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = supertrend(df, self.period, self.multiplier)
        entries = (direction == 1) & (direction.shift(1) == -1)
        exits   = (direction == -1) & (direction.shift(1) == 1)
        return entries, exits


class SupertrendLS(Strategy):
    """
    Long/short Supertrend — always in market, reverses on direction flip.
    """
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = supertrend(df, self.period, self.multiplier)
        to_bull = (direction == 1)  & (direction.shift(1) == -1)
        to_bear = (direction == -1) & (direction.shift(1) == 1)
        return to_bull, to_bear, to_bear, to_bull


class SupertrendRegime(Strategy):
    """
    Regime-filtered Supertrend:
    - Bull regime: long entries on uptrend flip
    - Bear regime: short entries on downtrend flip
    """
    def __init__(self, period=10, multiplier=3.0, regime_ma=200, regime_slope=20):
        self.period = period; self.multiplier = multiplier
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = supertrend(df, self.period, self.multiplier)
        reg  = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        to_bull = (direction == 1)  & (direction.shift(1) == -1)
        to_bear = (direction == -1) & (direction.shift(1) == 1)
        return to_bull & bull, to_bear, to_bear & bear, to_bull


# ─────────────────────────────────────────────────────────────────────────────
# Ichimoku
# ─────────────────────────────────────────────────────────────────────────────

class IchimokuStrategy(Strategy):
    """
    Long-only Ichimoku TK cross.
    Enter: Tenkan crosses above Kijun while price is above the cloud.
    Exit:  Tenkan crosses below Kijun.
    """
    def __init__(self, tenkan=9, kijun=26, senkou_b=52):
        self.tenkan = tenkan; self.kijun = kijun; self.senkou_b = senkou_b

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        ic = ichimoku(df, self.tenkan, self.kijun, self.senkou_b)
        tk = ic["tenkan_sen"]; kj = ic["kijun_sen"]
        sa = ic["senkou_a"];   sb = ic["senkou_b"]
        cloud_top    = pd.concat([sa, sb], axis=1).max(axis=1)
        above_cloud  = df["close"] > cloud_top
        tk_cross_up  = (tk > kj) & (tk.shift(1) <= kj.shift(1))
        tk_cross_dn  = (tk < kj) & (tk.shift(1) >= kj.shift(1))
        return tk_cross_up & above_cloud, tk_cross_dn


class IchimokuLS(Strategy):
    """
    Long/short Ichimoku TK cross (no regime filter).
    Long: TK cross up above cloud.
    Short: TK cross down below cloud.
    """
    def __init__(self, tenkan=9, kijun=26, senkou_b=52):
        self.tenkan = tenkan; self.kijun = kijun; self.senkou_b = senkou_b

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        ic = ichimoku(df, self.tenkan, self.kijun, self.senkou_b)
        tk = ic["tenkan_sen"]; kj = ic["kijun_sen"]
        sa = ic["senkou_a"];   sb = ic["senkou_b"]
        cloud_top    = pd.concat([sa, sb], axis=1).max(axis=1)
        cloud_bot    = pd.concat([sa, sb], axis=1).min(axis=1)
        above_cloud  = df["close"] > cloud_top
        below_cloud  = df["close"] < cloud_bot
        tk_cross_up  = (tk > kj) & (tk.shift(1) <= kj.shift(1))
        tk_cross_dn  = (tk < kj) & (tk.shift(1) >= kj.shift(1))
        return tk_cross_up & above_cloud, tk_cross_dn, tk_cross_dn & below_cloud, tk_cross_up


class IchimokuRegime(Strategy):
    """
    Regime-filtered Ichimoku TK cross.
    """
    def __init__(self, tenkan=9, kijun=26, senkou_b=52, regime_ma=200, regime_slope=20):
        self.tenkan = tenkan; self.kijun = kijun; self.senkou_b = senkou_b
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        ic = ichimoku(df, self.tenkan, self.kijun, self.senkou_b)
        tk = ic["tenkan_sen"]; kj = ic["kijun_sen"]
        sa = ic["senkou_a"];   sb = ic["senkou_b"]
        cloud_top   = pd.concat([sa, sb], axis=1).max(axis=1)
        cloud_bot   = pd.concat([sa, sb], axis=1).min(axis=1)
        above_cloud = df["close"] > cloud_top
        below_cloud = df["close"] < cloud_bot
        tk_up = (tk > kj) & (tk.shift(1) <= kj.shift(1))
        tk_dn = (tk < kj) & (tk.shift(1) >= kj.shift(1))
        reg  = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        return tk_up & above_cloud & bull, tk_dn, tk_dn & below_cloud & bear, tk_up


# ─────────────────────────────────────────────────────────────────────────────
# Williams %R
# ─────────────────────────────────────────────────────────────────────────────

class WilliamsRStrategy(Strategy):
    """
    Long-only Williams %R mean reversion.
    Enter: %R crosses above oversold threshold (-80).
    Exit:  %R crosses below overbought threshold (-20).
    """
    def __init__(self, period=14, oversold=-80, overbought=-20):
        self.period = period; self.oversold = oversold; self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        wr = williams_r(df, self.period)
        entries = (wr > self.oversold)   & (wr.shift(1) <= self.oversold)
        exits   = (wr < self.overbought) & (wr.shift(1) >= self.overbought)
        return entries, exits


class WilliamsRLS(Strategy):
    """
    Long/short Williams %R.
    """
    def __init__(self, period=14, oversold=-80, overbought=-20):
        self.period = period; self.oversold = oversold; self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        wr = williams_r(df, self.period)
        le = (wr > self.oversold)   & (wr.shift(1) <= self.oversold)
        lx = (wr < self.overbought) & (wr.shift(1) >= self.overbought)
        se = (wr < self.overbought) & (wr.shift(1) >= self.overbought)
        sx = (wr > self.oversold)   & (wr.shift(1) <= self.oversold)
        return le, lx, se, sx


class WilliamsRRegime(Strategy):
    """
    Regime-filtered Williams %R:
    Bull regime: buy oversold dips; Bear regime: sell overbought rallies.
    """
    def __init__(self, period=14, oversold=-80, overbought=-20, regime_ma=200, regime_slope=20):
        self.period = period; self.oversold = oversold; self.overbought = overbought
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        wr  = williams_r(df, self.period)
        reg = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        le = (wr > self.oversold)   & (wr.shift(1) <= self.oversold)  & bull
        lx = (wr < self.overbought) & (wr.shift(1) >= self.overbought)
        se = (wr < self.overbought) & (wr.shift(1) >= self.overbought) & bear
        sx = (wr > self.oversold)   & (wr.shift(1) <= self.oversold)
        return le, lx, se, sx


# ─────────────────────────────────────────────────────────────────────────────
# Donchian Channels
# ─────────────────────────────────────────────────────────────────────────────

class DonchianBreakout(Strategy):
    """
    Long-only Donchian channel breakout (Turtle Trading).
    Enter: close breaks above the N-bar upper channel.
    Exit:  close drops below the N-bar lower channel.
    """
    def __init__(self, period: int = 20):
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        upper, _, lower = donchian_channels(df, self.period)
        c = df["close"]
        # Use shift(1) so signal fires on the bar *after* the breakout bar
        entries = c > upper.shift(1)
        exits   = c < lower.shift(1)
        return entries, exits


class DonchianBreakoutLS(Strategy):
    """
    Long/short Donchian: buy upper breakout, short lower breakdown.
    """
    def __init__(self, period: int = 20):
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        upper, _, lower = donchian_channels(df, self.period)
        c = df["close"]
        le = c > upper.shift(1)
        lx = c < lower.shift(1)
        se = c < lower.shift(1)
        sx = c > upper.shift(1)
        return le, lx, se, sx


class DonchianBreakoutRegime(Strategy):
    """
    Regime-filtered Donchian:
    Bull: upper breakouts (trend-following longs).
    Bear: lower breakdowns (trend-following shorts).
    """
    def __init__(self, period=20, regime_ma=200, regime_slope=20):
        self.period = period
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        upper, _, lower = donchian_channels(df, self.period)
        c   = df["close"]
        reg = trend_regime(c, self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        le = (c > upper.shift(1)) & bull
        lx =  c < lower.shift(1)
        se = (c < lower.shift(1)) & bear
        sx =  c > upper.shift(1)
        return le, lx, se, sx


# ─────────────────────────────────────────────────────────────────────────────
# Parabolic SAR
# ─────────────────────────────────────────────────────────────────────────────

class PSARStrategy(Strategy):
    """
    Long-only Parabolic SAR.
    Enter: SAR direction flips from downtrend to uptrend (+1).
    Exit:  SAR direction flips from uptrend to downtrend (-1).
    """
    def __init__(self, initial_af=0.02, max_af=0.20, step_af=0.02):
        self.initial_af = initial_af
        self.max_af     = max_af
        self.step_af    = step_af

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = parabolic_sar(df, self.initial_af, self.max_af, self.step_af)
        entries = (direction == 1)  & (direction.shift(1) == -1)
        exits   = (direction == -1) & (direction.shift(1) == 1)
        return entries, exits


class PSARLS(Strategy):
    """
    Long/short Parabolic SAR — always in market, reverses on SAR flip.
    """
    def __init__(self, initial_af=0.02, max_af=0.20, step_af=0.02):
        self.initial_af = initial_af
        self.max_af     = max_af
        self.step_af    = step_af

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = parabolic_sar(df, self.initial_af, self.max_af, self.step_af)
        to_bull = (direction == 1)  & (direction.shift(1) == -1)
        to_bear = (direction == -1) & (direction.shift(1) == 1)
        return to_bull, to_bear, to_bear, to_bull


class PSARRegime(Strategy):
    """
    Regime-filtered PSAR:
    Bull: enter on uptrend flip; Bear: enter short on downtrend flip.
    """
    def __init__(self, initial_af=0.02, max_af=0.20, step_af=0.02,
                 regime_ma=200, regime_slope=20):
        self.initial_af = initial_af; self.max_af = max_af; self.step_af = step_af
        self.regime_ma = regime_ma; self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        _, direction = parabolic_sar(df, self.initial_af, self.max_af, self.step_af)
        reg  = trend_regime(df["close"], self.regime_ma, self.regime_slope)
        bull = reg == 1; bear = reg == -1
        to_bull = (direction == 1)  & (direction.shift(1) == -1)
        to_bear = (direction == -1) & (direction.shift(1) == 1)
        return to_bull & bull, to_bear, to_bear & bear, to_bull
