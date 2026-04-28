from backtest import Backtest
from strategy import (
    EMACrossover,        EMACrossoverLS,        EMACrossoverRegime,
    RSIMeanReversion,    RSIMeanReversionLS,    RSIMeanReversionRegime,
    BollingerBreakout,   BollingerBreakoutLS,   BollingerBreakoutRegime,
    MACDCrossover,       MACDCrossoverLS,        MACDCrossoverRegime,
)
from optimizer import GridOptimizer
import yfinance as yf

df = yf.download("^NSEI", start="2022-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

bt = Backtest(initial=100_000)


def show(label: str, m) -> None:
    print(f"  {label:<32} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | Trades {m.total_trades}")


# ── 3-way comparison ───────────────────────────────────────────────────────────
print("\n=== Long-only  vs  Long/Short  vs  Regime-Filtered ===")
trios = [
    ("EMA Crossover",
        EMACrossover(), EMACrossoverLS(), EMACrossoverRegime()),
    ("RSI Mean Reversion",
        RSIMeanReversion(), RSIMeanReversionLS(), RSIMeanReversionRegime()),
    ("Bollinger Breakout",
        BollingerBreakout(), BollingerBreakoutLS(), BollingerBreakoutRegime()),
    ("MACD Crossover",
        MACDCrossover(), MACDCrossoverLS(), MACDCrossoverRegime()),
]
for name, lo, ls, rg in trios:
    print(f"\n{name}")
    show("Long-only",        bt.run(df, lo)[2])
    show("Long/short",       bt.run(df, ls)[2])
    show("Regime-filtered",  bt.run(df, rg)[2])

# ── Grid Search: best regime-filtered strategy ──────────────────────────────
print("\n=== Grid Search: EMACrossoverRegime ===")
ema_opt = GridOptimizer(
    strategy_class=EMACrossoverRegime,
    param_grid={
        "fast":          [5, 8, 10, 12],
        "slow":          [20, 26, 30, 50],
        "regime_slope":  [10, 20, 30],
    },
    initial=100_000,
    min_trades=3,
    constraint=lambda p: p["fast"] < p["slow"],
)
ema_results = ema_opt.run(df, sort_by="sharpe_ratio")
print(ema_results.head(5).to_string(index=False))
print(f"\n{ema_opt.best(df)}")

print("\n=== Grid Search: RSIMeanReversionRegime ===")
rsi_opt = GridOptimizer(
    strategy_class=RSIMeanReversionRegime,
    param_grid={
        "period":        [10, 14, 20],
        "oversold":      [25, 30, 35],
        "overbought":    [65, 70, 75],
        "regime_slope":  [10, 20, 30],
    },
    initial=100_000,
    min_trades=3,
    constraint=lambda p: p["oversold"] < p["overbought"],
)
rsi_results = rsi_opt.run(df, sort_by="sharpe_ratio")
print(rsi_results.head(5).to_string(index=False))
print(f"\n{rsi_opt.best(df)}")
