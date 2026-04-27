
from backtest import Backtest
from strategy import EMACrossover
import yfinance as yf 

df = yf.download("^NSEI", start="2022-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

bt = Backtest(initial=100_000)

trades, equity, m = bt.run(df, EMACrossover())

print(f"Sharpe:    {m.sharpe_ratio:.2f}")
print(f"CAGR:      {m.cagr:.2%}")
print(f"Max DD:    {m.max_drawdown_pct:.2%}  ({m.max_drawdown_duration_days} days)")
print(f"Win Rate:  {m.win_rate:.2%}")
print(f"Trades:    {m.total_trades}")
print(f"Return:    {m.total_return_pct:.2%}")