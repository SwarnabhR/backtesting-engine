from backtest import Backtest
from strategy import (
    EMACrossover,        EMACrossoverLS,        EMACrossoverRegime,
    RSIMeanReversion,    RSIMeanReversionLS,    RSIMeanReversionRegime,
    BollingerBreakout,   BollingerBreakoutLS,   BollingerBreakoutRegime,
    MACDCrossover,       MACDCrossoverLS,        MACDCrossoverRegime,
)
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
from sizer import FixedSizer, PercentEquitySizer, ATRSizer, KellySizer
from indicators import atr
from plotter import plot_equity, plot_walk_forward, plot_optimizer
import yfinance as yf
from pathlib import Path

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# 6 years of data
df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]
df["atr"] = atr(df, period=14)

df3 = df["2022":"2024"]


def show(label: str, m) -> None:
    print(f"  {label:<38} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | Trades {m.total_trades}")


# ── 3-way comparison (2022–2024) ────────────────────────────────────────────────
print("\n=== Long-only vs Long/Short vs Regime-Filtered (2022–2024) ===")
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
bt = Backtest(initial=100_000)
for name, lo, ls, rg in trios:
    print(f"\n{name}")
    show("Long-only",       bt.run(df3, lo)[2])
    show("Long/short",      bt.run(df3, ls)[2])
    show("Regime-filtered", bt.run(df3, rg)[2])


# ── Equity curve plot: EMA Regime (best strategy, full 6y) ─────────────────
print("\n=== Equity Plot: EMA Regime (2019–2025) ===")
trades_ema, equity_ema, m_ema = bt.run(df, EMACrossoverRegime())
plot_equity(
    equity=equity_ema,
    trades=trades_ema,
    benchmark=df["close"],
    title="EMA Crossover — Regime Filtered (2019–2025)",
    save_path=OUT / "equity_ema_regime.png",
    initial=100_000,
)
show("EMA Regime (2019-2025)", m_ema)


# ── Position Sizing Comparison ──────────────────────────────────────────────────
print("\n=== Position Sizing: EMA Crossover (2022–2024) ===")
strat = EMACrossover()

bt1 = Backtest(initial=100_000, sizer=FixedSizer(shares=1))
trades1, equity1, _ = bt1.run(df3, strat)
show("FixedSizer(1)",       bt1.run(df3, strat)[2])

bt2 = Backtest(initial=100_000, sizer=PercentEquitySizer(pct=0.10))
show("PercentEquity(10%)",  bt2.run(df3, strat)[2])

bt3 = Backtest(initial=100_000, sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0))
trades3, equity3, _ = bt3.run(df3, strat)
show("ATRSizer(1%, 2xATR)", bt3.run(df3, strat)[2])

# Plot ATR-sized equity vs Fixed for comparison
plot_equity(
    equity=equity3,
    trades=trades3,
    benchmark=df3["close"],
    title="EMA Crossover — ATR Sizing 1% risk (2022–2024)",
    save_path=OUT / "equity_ema_atr.png",
    initial=100_000,
)

if not trades1.empty:
    kelly = KellySizer.from_trades(trades1, fraction=0.5)
    bt4 = Backtest(initial=100_000, sizer=kelly)
    show(f"KellySizer({kelly})", bt4.run(df3, strat)[2])


# ── Grid Optimizer ─────────────────────────────────────────────────────────────
print("\n=== Optimizer: EMA Crossover (2022–2024) ===")
opt = GridOptimizer(
    strategy_class=EMACrossover,
    param_grid={
        "fast": [5, 8, 10, 12, 15, 20],
        "slow": [20, 26, 30, 40, 50, 60],
    },
    initial=100_000,
    min_trades=3,
    constraint=lambda p: p["fast"] < p["slow"],
)
opt_results = opt.run(df3, sort_by="sharpe_ratio")
print(opt_results.head(5).to_string(index=False))

plot_optimizer(
    results_df=opt_results,
    x="fast",
    y="slow",
    color_metric="sharpe_ratio",
    top_n=5,
    title="EMA Crossover — Grid Search Sharpe (2022–2024)",
    save_path=OUT / "optimizer_ema.png",
)


# ── Walk-Forward: EMACrossover — anchored ────────────────────────────────
print("\n=== Walk-Forward: EMACrossover (anchored, 2-yr IS / 1-yr OOS) ===")
wfv_ema = WalkForwardValidator(
    strategy_class=EMACrossover,
    param_grid={"fast": [5, 8, 10, 12], "slow": [20, 26, 30, 50]},
    is_bars=504, oos_bars=252, warmup_bars=0,
    anchored=True, sort_by="sharpe_ratio",
    initial=100_000, min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
wf_ema = wfv_ema.run(df)
print(wf_ema.summary.to_string(index=False))
print(f"\n{wf_ema}")

plot_walk_forward(
    wf_result=wf_ema,
    title="Walk-Forward: EMA Crossover (anchored)",
    save_path=OUT / "wfv_ema.png",
)


# ── Walk-Forward: EMACrossoverRegime — rolling, warmup_bars=200 ───────────
print("\n=== Walk-Forward: EMACrossoverRegime (rolling, 1.5-yr IS / 1-yr OOS) ===")
wfv_regime = WalkForwardValidator(
    strategy_class=EMACrossoverRegime,
    param_grid={
        "fast":         [5, 8, 12],
        "slow":         [20, 26, 50],
        "regime_slope": [10, 20, 30],
    },
    is_bars=378, oos_bars=252, warmup_bars=200,
    anchored=False, sort_by="sharpe_ratio",
    initial=100_000, min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
wf_regime = wfv_regime.run(df)
print(wf_regime.summary.to_string(index=False))
print(f"\n{wf_regime}")

plot_walk_forward(
    wf_result=wf_regime,
    title="Walk-Forward: EMA Regime (rolling, warmup=200)",
    save_path=OUT / "wfv_ema_regime.png",
)


# ── Walk-Forward: RSIMeanReversionRegime — anchored ─────────────────────
print("\n=== Walk-Forward: RSIMeanReversionRegime (anchored, 2-yr IS / 1-yr OOS) ===")
wfv_rsi = WalkForwardValidator(
    strategy_class=RSIMeanReversionRegime,
    param_grid={
        "period":       [10, 14, 20],
        "oversold":     [25, 30, 35],
        "overbought":   [65, 70, 75],
        "regime_slope": [10, 20],
    },
    is_bars=504, oos_bars=252, warmup_bars=200,
    anchored=True, sort_by="sharpe_ratio",
    initial=100_000, min_trades=2,
    constraint=lambda p: p["oversold"] < p["overbought"],
)
wf_rsi = wfv_rsi.run(df)
print(wf_rsi.summary.to_string(index=False))
print(f"\n{wf_rsi}")

plot_walk_forward(
    wf_result=wf_rsi,
    title="Walk-Forward: RSI Regime (anchored, warmup=200)",
    save_path=OUT / "wfv_rsi_regime.png",
)

print(f"\n✔ Charts saved to {OUT.resolve()}/")
