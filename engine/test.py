from backtest import Backtest
from strategy import (
    EMACrossover,        EMACrossoverLS,        EMACrossoverRegime,
    RSIMeanReversion,    RSIMeanReversionLS,    RSIMeanReversionRegime,
    BollingerBreakout,   BollingerBreakoutLS,   BollingerBreakoutRegime,
    MACDCrossover,       MACDCrossoverLS,        MACDCrossoverRegime,
)
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
import yfinance as yf

# 6 years: enough for multiple WFV folds even with 200-bar warmup
df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

df3 = df["2022":"2024"]   # 3-year slice for quick comparison table
bt  = Backtest(initial=100_000)


def show(label: str, m) -> None:
    print(f"  {label:<32} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | Trades {m.total_trades}")


# ── 3-way comparison (2022–2024) ────────────────────────────────────────────────
print("\n=== Long-only  vs  Long/Short  vs  Regime-Filtered (2022–2024) ===")
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
    show("Long-only",       bt.run(df3, lo)[2])
    show("Long/short",      bt.run(df3, ls)[2])
    show("Regime-filtered", bt.run(df3, rg)[2])


# ── Walk-Forward: EMACrossover — anchored, no warmup needed ────────────────
print("\n=== Walk-Forward: EMACrossover (anchored, 2-yr IS / 1-yr OOS) ===")
wfv_ema = WalkForwardValidator(
    strategy_class=EMACrossover,
    param_grid={"fast": [5, 8, 10, 12], "slow": [20, 26, 30, 50]},
    is_bars=504,      # ~2 trading years
    oos_bars=252,     # ~1 trading year  ← raised from 6mo; more OOS trades
    warmup_bars=0,
    anchored=True,
    sort_by="sharpe_ratio",
    initial=100_000,
    min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
wf_ema = wfv_ema.run(df)
print(wf_ema.summary.to_string(index=False))
print(f"\n{wf_ema}")


# ── Walk-Forward: EMACrossoverRegime — rolling, warmup_bars=200 ────────────
print("\n=== Walk-Forward: EMACrossoverRegime (rolling, 1.5-yr IS / 1-yr OOS) ===")
wfv_regime = WalkForwardValidator(
    strategy_class=EMACrossoverRegime,
    param_grid={
        "fast":         [5, 8, 12],
        "slow":         [20, 26, 50],
        "regime_slope": [10, 20, 30],
    },
    is_bars=378,      # ~1.5 yr of usable bars (after warmup is accounted for)
    oos_bars=252,     # ~1 yr
    warmup_bars=200,  # 200-bar SMA needs 200 bars before first valid value
    anchored=False,
    sort_by="sharpe_ratio",
    initial=100_000,
    min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
wf_regime = wfv_regime.run(df)
print(wf_regime.summary.to_string(index=False))
print(f"\n{wf_regime}")


# ── Walk-Forward: RSIMeanReversionRegime — anchored, warmup_bars=200 ───────
print("\n=== Walk-Forward: RSIMeanReversionRegime (anchored, 2-yr IS / 1-yr OOS) ===")
wfv_rsi = WalkForwardValidator(
    strategy_class=RSIMeanReversionRegime,
    param_grid={
        "period":       [10, 14, 20],
        "oversold":     [25, 30, 35],
        "overbought":   [65, 70, 75],
        "regime_slope": [10, 20],
    },
    is_bars=504,
    oos_bars=252,
    warmup_bars=200,
    anchored=True,
    sort_by="sharpe_ratio",
    initial=100_000,
    min_trades=2,
    constraint=lambda p: p["oversold"] < p["overbought"],
)
wf_rsi = wfv_rsi.run(df)
print(wf_rsi.summary.to_string(index=False))
print(f"\n{wf_rsi}")
