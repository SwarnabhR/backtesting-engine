
from backtest import Backtest
from strategy import EMACrossover
import yfinance as yf 

df = yf.download("RELIANCE.NS", start="2024-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

bt = Backtest(initial=100_000)

trades, equity, m = bt.run(df, EMACrossover())

print(f"{m.sharpe_ratio:.2f}")
print(f"{m.cagr}")
print(f"{m.max_drawdown_pct}")
print(f"{m.win_rate}")
print(f"{m.total_trades}")
print(f"{m.total_return_pct}")
