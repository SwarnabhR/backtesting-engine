from abc import ABC, abstractmethod
import pandas as pd
from indicators import ema, rsi, bollinger_bands, macd


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
    """
    Long/short: golden cross → go long; death cross → go short.
    Each cross simultaneously exits the opposite leg.
    """

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)

        golden_cross = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        death_cross  = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        long_entries  = golden_cross
        long_exits    = death_cross    # death cross exits long AND enters short
        short_entries = death_cross
        short_exits   = golden_cross   # golden cross exits short AND enters long

        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# RSI Mean Reversion
# ────────────────────────────────────────────────────────────────────────────

class RSIMeanReversion(Strategy):
    """Long-only: buy RSI crossing up through oversold, sell at overbought."""

    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        rsi_vals = rsi(close, self.period)
        entries = (rsi_vals > self.oversold)    & (rsi_vals.shift(1) <= self.oversold)
        exits   = (rsi_vals > self.overbought)  & (rsi_vals.shift(1) <= self.overbought)
        return entries, exits


class RSIMeanReversionLS(Strategy):
    """
    Long/short mean-reversion:
      - Long  when RSI crosses UP  through oversold threshold
      - Short when RSI crosses DOWN through overbought threshold
      - Exit long  at overbought; exit short at oversold
    """

    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        rsi_vals = rsi(close, self.period)

        long_entries  = (rsi_vals > self.oversold)   & (rsi_vals.shift(1) <= self.oversold)
        long_exits    = (rsi_vals > self.overbought)  & (rsi_vals.shift(1) <= self.overbought)
        short_entries = (rsi_vals < self.overbought)  & (rsi_vals.shift(1) >= self.overbought)
        short_exits   = (rsi_vals < self.oversold)    & (rsi_vals.shift(1) >= self.oversold)

        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ────────────────────────────────────────────────────────────────────────────

class BollingerBreakout(Strategy):
    """Long-only: buy on upper-band breakout, sell on mean reversion to middle."""

    def __init__(self, period: int = 20, std_dev: int = 2):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)
        entries = (close > upper)   & (close.shift(1) <= upper.shift(1))
        exits   = (close < middle)  & (close.shift(1) >= middle.shift(1))
        return entries, exits


class BollingerBreakoutLS(Strategy):
    """
    Long/short Bollinger:
      - Long  on upper-band breakout  → exit when price falls back to middle
      - Short on lower-band breakdown → exit when price rises back to middle
    """

    def __init__(self, period: int = 20, std_dev: int = 2):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df["close"]
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)

        long_entries  = (close > upper)   & (close.shift(1) <= upper.shift(1))
        long_exits    = (close < middle)  & (close.shift(1) >= middle.shift(1))
        short_entries = (close < lower)   & (close.shift(1) >= lower.shift(1))
        short_exits   = (close > middle)  & (close.shift(1) <= middle.shift(1))

        return long_entries, long_exits, short_entries, short_exits


# ────────────────────────────────────────────────────────────────────────────
# MACD Crossover (new — indicator already existed in indicators.py)
# ────────────────────────────────────────────────────────────────────────────

class MACDCrossover(Strategy):
    """
    Long-only MACD signal-line crossover.
    Buy when MACD line crosses above signal line;
    sell when it crosses below.
    """

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
    """
    Long/short MACD:
      - Long  when MACD crosses above signal line
      - Short when MACD crosses below signal line
    """

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
