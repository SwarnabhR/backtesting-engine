from backtest import Backtest
from strategy import EMACrossover, RSIMeanReversion, BollingerBreakout
from optimizer import GridOptimizer
import yfinance as yf

df = yf.download("^NSEI", start="2022-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

# ── Baseline: fixed-parameter strategies ────────────────────────────────────
print("\n=== Baseline (default params) ===")
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

# ── Grid Search: EMA Crossover ───────────────────────────────────────────────
print("\n=== Grid Search: EMA Crossover ===")
ema_opt = GridOptimizer(
    strategy_class=EMACrossover,
    param_grid={
        "fast": [5, 8, 10, 12, 15],
        "slow": [20, 26, 30, 40, 50],
    },
    initial=100_000,
    min_trades=3,
    constraint=lambda p: p["fast"] < p["slow"],  # enforce fast < slow
)
ema_results = ema_opt.run(df, sort_by="sharpe_ratio")
print(ema_results.head(5).to_string(index=False))
best_ema = ema_opt.best(df, sort_by="sharpe_ratio")
if best_ema:
    print(f"\n{best_ema}")

# ── Grid Search: RSI Mean Reversion ─────────────────────────────────────────
print("\n=== Grid Search: RSI Mean Reversion ===")
rsi_opt = GridOptimizer(
    strategy_class=RSIMeanReversion,
    param_grid={
        "period": [10, 14, 20],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
    },
    initial=100_000,
    min_trades=3,
    constraint=lambda p: p["oversold"] < p["overbought"],  # logical guard
)
rsi_results = rsi_opt.run(df, sort_by="cagr")
print(rsi_results.head(5).to_string(index=False))
best_rsi = rsi_opt.best(df, sort_by="cagr")
if best_rsi:
    print(f"\n{best_rsi}")

# ── Grid Search: Bollinger Breakout ─────────────────────────────────────────
print("\n=== Grid Search: Bollinger Breakout ===")
bb_opt = GridOptimizer(
    strategy_class=BollingerBreakout,
    param_grid={
        "period": [15, 20, 25, 30],
        "std_dev": [1, 2, 3],
    },
    initial=100_000,
    min_trades=3,
)
bb_results = bb_opt.run(df, sort_by="sharpe_ratio")
print(bb_results.head(5).to_string(index=False))
best_bb = bb_opt.best(df, sort_by="sharpe_ratio")
if best_bb:
    print(f"\n{best_bb}")
