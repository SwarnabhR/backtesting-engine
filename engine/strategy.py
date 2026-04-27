from abc import ABC, abstractmethod
import pandas as pd  
from indicators import ema, rsi, bollinger_bands

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        pass
    
class EMACrossover(Strategy):
    def __init__(self, fast: int=12, slow: int=26):
        self.fast = fast
        self.slow = slow
        
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df['close']
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        
        entries = (fast_ema > slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        return entries, exits
    
class RSIMeanReversion(Strategy):
    def __init__(self, oversold: int=30, overbought: int=70, period: int=14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period
        
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df['close']
        rsi_vals = rsi(close, self.period)
        entries = (rsi_vals > self.oversold) & (rsi_vals.shift(1) <= self.oversold)
        exits = (rsi_vals > self.overbought) & (rsi_vals.shift(1) <= self.overbought)
        
        return entries, exits
    
class BollingerBreakout(Strategy):
    def __init__(self, period: int=20, std_dev: int=2):
        self.period = period
        self.std_dev = std_dev
    def generate_signals(self, df: pd.DataFrame) -> tuple:
        close = df['close']
        upper, middle, lower = bollinger_bands(close, self.period, self.std_dev)
        
        entries = (close > upper) & (close.shift(1) <= upper.shift(1))
        exits = (close < middle) & (close.shift(1) >= middle.shift(1))
        
        return entries, exits
    
