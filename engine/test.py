
from backtest import Backtest
from strategy import EMACrossover, RSIMeanReversion, BollingerBreakout
import yfinance as yf 

df = yf.download("^NSEI", start="2022-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

bt = Backtest(initial=100_000)

strategies = [
    ("EMA Crossover",      EMACrossover()),
    ("RSI Mean Reversion", RSIMeanReversion()),
    ("Bollinger Breakout", BollingerBreakout()),
]
for name, strat in strategies:
    trades, equity, m = bt.run(df, strat)
    print(f"\n{name}")
    print(f"  Sharpe {m.sharpe_ratio:.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | "
          f"Trades {m.total_trades}")