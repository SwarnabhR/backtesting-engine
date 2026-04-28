from abc import ABC, abstractmethod
import pandas as pd
from indicators import ema, rsi, bollinger_bands, macd, trend_regime


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Return either:
          - 2-tuple (entries, exits)                          → long-only
          - 4-tuple (long_entries, long_exits,
                     short_entries, short_exits)              → long/short
        """
        pass


# ────────────────────────────────────────────────────────────────────────────
# EMA Crossover
# ────────────────────────────────────────────────────────────────────────────

class EMACrossover(Strategy):
    """Long-only: buy on golden cross, sell on death cross."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        exits   = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        return entries, exits


class EMACrossoverLS(Strategy):
    """Long/short (no filter): golden cross → long, death cross → short."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        golden = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        death  = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        return golden, death, death, golden


class EMACrossoverRegime(Strategy):
    """
    Regime-filtered EMA crossover.

    - Bull regime  (↑ 200-SMA slope): long entries allowed, shorts suppressed.
    - Bear regime  (↓ 200-SMA slope): short entries allowed, longs suppressed.
    - Neutral/warmup: no new entries in either direction.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        regime_ma: int = 200,
        regime_slope: int = 20,
    ):
        self.fast = fast
        self.slow = slow
        self.regime_ma = regime_ma
        self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        golden = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        death  = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        regime = trend_regime(close, self.regime_ma, self.regime_slope)
        bull = regime == 1
        bear = regime == -1

        long_entries  = golden & bull
        long_exits    = death
        short_entries = death  & bear
        short_exits   = golden
        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# RSI Mean Reversion
# ────────────────────────────────────────────────────────────────────────────

class RSIMeanReversion(Strategy):
    """Long-only RSI mean reversion."""

    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        rsi_vals = rsi(close, self.period)
        entries = (rsi_vals > self.oversold)   & (rsi_vals.shift(1) <= self.oversold)
        exits   = (rsi_vals > self.overbought) & (rsi_vals.shift(1) <= self.overbought)
        return entries, exits


class RSIMeanReversionLS(Strategy):
    """Long/short RSI (no regime filter)."""

    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        rsi_vals = rsi(close, self.period)
        long_entries  = (rsi_vals > self.oversold)   & (rsi_vals.shift(1) <= self.oversold)
        long_exits    = (rsi_vals > self.overbought) & (rsi_vals.shift(1) <= self.overbought)
        short_entries = (rsi_vals < self.overbought) & (rsi_vals.shift(1) >= self.overbought)
        short_exits   = (rsi_vals < self.oversold)   & (rsi_vals.shift(1) >= self.oversold)
        return long_entries, long_exits, short_entries, short_exits


class RSIMeanReversionRegime(Strategy):
    """
    Regime-filtered RSI:
    - Bull: only long entries (buy oversold dips in an uptrend)
    - Bear: only short entries (sell overbought rallies in a downtrend)
    """

    def __init__(
        self,
        oversold: int = 30,
        overbought: int = 70,
        period: int = 14,
        regime_ma: int = 200,
        regime_slope: int = 20,
    ):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period
        self.regime_ma = regime_ma
        self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        rsi_vals = rsi(close, self.period)
        regime = trend_regime(close, self.regime_ma, self.regime_slope)
        bull = regime == 1
        bear = regime == -1

        long_entries  = (rsi_vals > self.oversold)   & (rsi_vals.shift(1) <= self.oversold)  & bull
        long_exits    = (rsi_vals > self.overbought) & (rsi_vals.shift(1) <= self.overbought)
        short_entries = (rsi_vals < self.overbought) & (rsi_vals.shift(1) >= self.overbought) & bear
        short_exits   = (rsi_vals < self.oversold)   & (rsi_vals.shift(1) >= self.oversold)
        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ────────────────────────────────────────────────────────────────────────────

class BollingerBreakout(Strategy):
    """Long-only Bollinger breakout."""

    def __init__(self, period: int = 20, std_dev: int = 2):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)
        entries = (close > upper)  & (close.shift(1) <= upper.shift(1))
        exits   = (close < middle) & (close.shift(1) >= middle.shift(1))
        return entries, exits


class BollingerBreakoutLS(Strategy):
    """Long/short Bollinger (no regime filter)."""

    def __init__(self, period: int = 20, std_dev: int = 2):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)
        long_entries  = (close > upper)  & (close.shift(1) <= upper.shift(1))
        long_exits    = (close < middle) & (close.shift(1) >= middle.shift(1))
        short_entries = (close < lower)  & (close.shift(1) >= lower.shift(1))
        short_exits   = (close > middle) & (close.shift(1) <= middle.shift(1))
        return long_entries, long_exits, short_entries, short_exits


class BollingerBreakoutRegime(Strategy):
    """
    Regime-filtered Bollinger:
    - Bull: upper breakouts only (trend continuation longs)
    - Bear: lower breakdowns only (trend continuation shorts)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: int = 2,
        regime_ma: int = 200,
        regime_slope: int = 20,
    ):
        self.period = period
        self.std_dev = std_dev
        self.regime_ma = regime_ma
        self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)
        regime = trend_regime(close, self.regime_ma, self.regime_slope)
        bull = regime == 1
        bear = regime == -1

        long_entries  = (close > upper)  & (close.shift(1) <= upper.shift(1))  & bull
        long_exits    = (close < middle) & (close.shift(1) >= middle.shift(1))
        short_entries = (close < lower)  & (close.shift(1) >= lower.shift(1))  & bear
        short_exits   = (close > middle) & (close.shift(1) <= middle.shift(1))
        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# MACD Crossover
# ────────────────────────────────────────────────────────────────────────────

class MACDCrossover(Strategy):
    """Long-only MACD signal-line crossover."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        macd_line, signal_line, _ = macd(close, self.fast, self.slow, self.signal)
        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        exits   = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        return entries, exits


class MACDCrossoverLS(Strategy):
    """Long/short MACD (no regime filter)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        macd_line, signal_line, _ = macd(close, self.fast, self.slow, self.signal)
        long_entries  = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        long_exits    = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        short_entries = long_exits
        short_exits   = long_entries
        return long_entries, long_exits, short_entries, short_exits


class MACDCrossoverRegime(Strategy):
    """
    Regime-filtered MACD:
    - Bull: MACD bullish crossovers only
    - Bear: MACD bearish crossovers (shorts) only
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        regime_ma: int = 200,
        regime_slope: int = 20,
    ):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.regime_ma = regime_ma
        self.regime_slope = regime_slope

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        macd_line, signal_line, _ = macd(close, self.fast, self.slow, self.signal)
        regime = trend_regime(close, self.regime_ma, self.regime_slope)
        bull = regime == 1
        bear = regime == -1

        bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        long_entries  = bullish_cross & bull
        long_exits    = bearish_cross
        short_entries = bearish_cross & bear
        short_exits   = bullish_cross
        return long_entries, long_exits, short_entries, short_exits
