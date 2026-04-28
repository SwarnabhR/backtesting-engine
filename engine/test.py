from backtest import Backtest
from strategy import (
    EMACrossover,      EMACrossoverLS,
    RSIMeanReversion,  RSIMeanReversionLS,
    BollingerBreakout, BollingerBreakoutLS,
    MACDCrossover,     MACDCrossoverLS,
)
from optimizer import GridOptimizer
import yfinance as yf

df = yf.download("^NSEI", start="2022-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

bt = Backtest(initial=100_000)


def show(name: str, m) -> None:
    print(f"  {name:<30} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | Trades {m.total_trades}")


# ── Long-only vs Long/Short comparison ──────────────────────────────────
print("\n=== Long-only vs Long/Short ===")
pairs = [
    ("EMA Crossover",      EMACrossover(),      EMACrossoverLS()),
    ("RSI Mean Reversion", RSIMeanReversion(),  RSIMeanReversionLS()),
    ("Bollinger Breakout", BollingerBreakout(), BollingerBreakoutLS()),
    ("MACD Crossover",     MACDCrossover(),     MACDCrossoverLS()),
]
for name, lo, ls in pairs:
    print(f"\n{name}")
    _, _, m_lo = bt.run(df, lo)
    _, _, m_ls = bt.run(df, ls)
    show("Long-only", m_lo)
    show("Long/short", m_ls)

# ── Grid Search: EMA LS (optimise the long/short variant) ─────────────────
print("\n=== Grid Search: EMACrossoverLS ===")
ema_opt = GridOptimizer(
    strategy_class=EMACrossoverLS,
    param_grid={"fast": [5, 8, 10, 12, 15], "slow": [20, 26, 30, 40, 50]},
    initial=100_000,
    min_trades=5,
    constraint=lambda p: p["fast"] < p["slow"],
)
ema_results = ema_opt.run(df, sort_by="sharpe_ratio")
print(ema_results.head(5).to_string(index=False))
print(f"\n{ema_opt.best(df, sort_by='sharpe_ratio')}")

# ── Grid Search: MACD LS ──────────────────────────────────────────────────
print("\n=== Grid Search: MACDCrossoverLS ===")
macd_opt = GridOptimizer(
    strategy_class=MACDCrossoverLS,
    param_grid={
        "fast":   [8, 12, 15],
        "slow":   [21, 26, 30],
        "signal": [7, 9, 12],
    },
    initial=100_000,
    min_trades=5,
    constraint=lambda p: p["fast"] < p["slow"],
)
macd_results = macd_opt.run(df, sort_by="sharpe_ratio")
print(macd_results.head(5).to_string(index=False))
print(f"\n{macd_opt.best(df, sort_by='sharpe_ratio')}")
