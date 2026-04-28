from backtest import Backtest
from strategy import (
    EMACrossover,        EMACrossoverLS,        EMACrossoverRegime,
    RSIMeanReversion,    RSIMeanReversionLS,    RSIMeanReversionRegime,
    BollingerBreakout,   BollingerBreakoutLS,   BollingerBreakoutRegime,
    MACDCrossover,       MACDCrossoverLS,       MACDCrossoverRegime,
    SupertrendStrategy,  SupertrendLS,          SupertrendRegime,
    IchimokuStrategy,    IchimokuLS,            IchimokuRegime,
    DonchianBreakout,    DonchianBreakoutLS,    DonchianBreakoutRegime,
    PSARStrategy,        PSARLS,                PSARRegime,
    WilliamsRStrategy,   WilliamsRLS,           WilliamsRRegime,
)
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
from sizer import FixedSizer, PercentEquitySizer, ATRSizer, KellySizer
from costs import nse_equity_intraday, nse_equity_delivery
from indicators import atr
from portfolio import (
    PortfolioBacktest,
    EqualWeightSizer,
    FixedWeightSizer,
    VolTargetSizer,
    run_portfolio,
)
from regime_sizer import (
    RegimeEqualSizer,
    RegimeVolSizer,
    RegimeFixedSizer,
    slope_regime,
    supertrend_regime,
)
from plotter import plot_equity, plot_walk_forward, plot_optimizer
import yfinance as yf
from functools import partial
from pathlib import Path

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Downloading data ...")
df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]
df["atr"] = atr(df, period=14)
df3 = df["2022":"2024"]

# Four NSE symbols for portfolio tests
SYMBOLS = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS"]
raw_dfs = {}
for sym in SYMBOLS:
    tmp = yf.download(sym, start="2019-01-01", end="2025-01-01")
    tmp.columns = [c[0].lower() for c in tmp.columns]
    tmp["atr"] = atr(tmp, period=14)
    raw_dfs[sym] = tmp


def show(label: str, m) -> None:
    print(f"  {label:<44} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | WR {m.win_rate:.2%} | Trades {m.total_trades}")


def show_port(label: str, result) -> None:
    m = result.metrics
    print(f"  {label:<44} Sharpe {m.sharpe_ratio:+.2f} | CAGR {m.cagr:.2%} | "
          f"DD {m.max_drawdown_pct:.2%} | Trades {m.total_trades}")


# ===========================================================================
# 1. Single-asset: 3-way comparison (2022-2024)
# ===========================================================================
print("\n=== 1. Long-only vs Long/Short vs Regime-Filtered (2022-2024) ===")
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
    print(f"\n  {name}")
    show("    Long-only",       bt.run(df3, lo)[2])
    show("    Long/short",      bt.run(df3, ls)[2])
    show("    Regime-filtered", bt.run(df3, rg)[2])


# ===========================================================================
# 2. New indicators — 5 strategy families (2022-2024)
# ===========================================================================
print("\n=== 2. New Indicator Strategies (2022-2024) ===")
new_trios = [
    ("Supertrend",
        SupertrendStrategy(), SupertrendLS(), SupertrendRegime()),
    ("Ichimoku",
        IchimokuStrategy(), IchimokuLS(), IchimokuRegime()),
    ("Donchian Breakout",
        DonchianBreakout(), DonchianBreakoutLS(), DonchianBreakoutRegime()),
    ("Parabolic SAR",
        PSARStrategy(), PSARLS(), PSARRegime()),
    ("Williams %R",
        WilliamsRStrategy(), WilliamsRLS(), WilliamsRRegime()),
]
for name, lo, ls, rg in new_trios:
    print(f"\n  {name}")
    show("    Long-only",       bt.run(df3, lo)[2])
    show("    Long/short",      bt.run(df3, ls)[2])
    show("    Regime-filtered", bt.run(df3, rg)[2])


# ===========================================================================
# 3. Equity curve plot: EMA Regime (full 6y)
# ===========================================================================
print("\n=== 3. Equity Plot: EMA Regime (2019-2025) ===")
trades_ema, equity_ema, m_ema = bt.run(df, EMACrossoverRegime())
plot_equity(
    equity=equity_ema,
    trades=trades_ema,
    benchmark=df["close"],
    title="EMA Crossover — Regime Filtered (2019-2025)",
    save_path=OUT / "equity_ema_regime.png",
    initial=100_000,
)
show("EMA Regime (2019-2025)", m_ema)


# ===========================================================================
# 4. Position Sizing Comparison
# ===========================================================================
print("\n=== 4. Position Sizing: EMA Crossover (2022-2024) ===")
strat = EMACrossover()
bt1 = Backtest(initial=100_000, sizer=FixedSizer(shares=1))
trades1, equity1, _ = bt1.run(df3, strat)
show("FixedSizer(1)",       bt1.run(df3, strat)[2])

bt2 = Backtest(initial=100_000, sizer=PercentEquitySizer(pct=0.10))
show("PercentEquity(10%)",  bt2.run(df3, strat)[2])

bt3 = Backtest(initial=100_000, sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0))
trades3, equity3, _ = bt3.run(df3, strat)
show("ATRSizer(1%, 2xATR)", bt3.run(df3, strat)[2])

plot_equity(
    equity=equity3,
    trades=trades3,
    benchmark=df3["close"],
    title="EMA Crossover — ATR Sizing 1% risk (2022-2024)",
    save_path=OUT / "equity_ema_atr.png",
    initial=100_000,
)

if not trades1.empty:
    kelly = KellySizer.from_trades(trades1, fraction=0.5)
    bt4 = Backtest(initial=100_000, sizer=kelly)
    show(f"KellySizer({kelly})", bt4.run(df3, strat)[2])


# ===========================================================================
# 5. Transaction Costs
# ===========================================================================
print("\n=== 5. Transaction Costs: EMA Crossover (2022-2024) ===")
bt_nc = Backtest(initial=100_000)
bt_id = Backtest(initial=100_000, cost_model=nse_equity_intraday())
bt_dv = Backtest(initial=100_000, cost_model=nse_equity_delivery())
show("No cost",         bt_nc.run(df3, EMACrossover())[2])
show("NSE Intraday",    bt_id.run(df3, EMACrossover())[2])
show("NSE Delivery",    bt_dv.run(df3, EMACrossover())[2])


# ===========================================================================
# 6. Grid Optimizer
# ===========================================================================
print("\n=== 6. Optimizer: EMA Crossover (2022-2024) ===")
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
    x="fast", y="slow", color_metric="sharpe_ratio",
    top_n=5,
    title="EMA Crossover — Grid Search Sharpe (2022-2024)",
    save_path=OUT / "optimizer_ema.png",
)


# ===========================================================================
# 7. Walk-Forward Validation
# ===========================================================================
print("\n=== 7a. Walk-Forward: EMACrossover (anchored, 2-yr IS / 1-yr OOS) ===")
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

print("\n=== 7b. Walk-Forward: EMACrossoverRegime (rolling, 1.5-yr IS / 1-yr OOS) ===")
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

print("\n=== 7c. Walk-Forward: RSIMeanReversionRegime (anchored) ===")
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


# ===========================================================================
# 8. Multi-Asset Portfolio: basic allocation modes
# ===========================================================================
print("\n=== 8. Portfolio Backtest: 4 NSE symbols (2019-2025) ===")

# 8a. Equal weight
pb_eq = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=EqualWeightSizer(),
    cost_model=nse_equity_delivery(),
)
result_eq = pb_eq.run(raw_dfs, EMACrossoverRegime)
show_port("EqualWeight — EMA Regime", result_eq)
for sym, m in result_eq.symbol_metrics.items():
    show(f"    {sym}", m)

# 8b. Fixed weight (conviction)
pb_fw = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=FixedWeightSizer({
        "RELIANCE.NS": 0.35,
        "INFY.NS":     0.25,
        "HDFCBANK.NS": 0.25,
        "TCS.NS":      0.15,
    }),
)
result_fw = pb_fw.run(raw_dfs, EMACrossoverRegime)
show_port("FixedWeight — EMA Regime", result_fw)

# 8c. Inverse-volatility (monthly rebalance)
pb_vt = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=VolTargetSizer(lookback=20, rebalance_bars=21, cap=0.40),
)
result_vt = pb_vt.run(raw_dfs, SupertrendRegime)
show_port("VolTarget (monthly) — Supertrend Regime", result_vt)

# 8d. One-liner
result_ol = run_portfolio(
    raw_dfs, SupertrendRegime,
    initial=1_000_000,
    cost_model=nse_equity_delivery(),
)
show_port("run_portfolio() one-liner — Supertrend", result_ol)


# ===========================================================================
# 9. Regime-Aware Portfolio Sizers
# ===========================================================================
print("\n=== 9. Regime-Aware Portfolio Sizers: 4 NSE symbols (2019-2025) ===")

slope50   = partial(slope_regime, period=50)
st_regime = partial(supertrend_regime, period=10, multiplier=3.0)

# 9a. RegimeEqualSizer — slope regime
pb_re = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=RegimeEqualSizer(regime_fn=slope50),
    cost_model=nse_equity_delivery(),
)
result_re = pb_re.run(raw_dfs, EMACrossoverRegime)
show_port("RegimeEqual (slope50) — EMA Regime", result_re)

# 9b. RegimeVolSizer — supertrend regime, monthly rebalance
pb_rv = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=RegimeVolSizer(
        regime_fn=st_regime, lookback=20, rebalance_bars=21, cap=0.40
    ),
    cost_model=nse_equity_delivery(),
)
result_rv = pb_rv.run(raw_dfs, SupertrendRegime)
show_port("RegimeVol (ST regime, monthly) — Supertrend", result_rv)

# 9c. RegimeFixedSizer — user conviction weights, slope-gated
pb_rf = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=RegimeFixedSizer(
        weights={
            "RELIANCE.NS": 0.35,
            "INFY.NS":     0.25,
            "HDFCBANK.NS": 0.25,
            "TCS.NS":      0.15,
        },
        regime_fn=slope50,
    ),
)
result_rf = pb_rf.run(raw_dfs, EMACrossoverRegime)
show_port("RegimeFixed (slope50) — EMA Regime", result_rf)

# 9d. Side-by-side: VolTarget vs RegimeVol on same strategy
print("\n  Side-by-side: VolTarget vs RegimeVol — SupertrendRegime")
show_port("  VolTargetSizer (no regime gate)", result_vt)
show_port("  RegimeVolSizer (ST regime gate)", result_rv)


print(f"\n✔ All sections complete. Charts saved to {OUT.resolve()}/")
