# backtesting-engine

A modular, from-scratch backtesting engine for systematic trading strategies.
Built in pure Python/Pandas — no `backtrader`, no `vectorbt`.

---

## Architecture

```
engine/
├── data.py            # yfinance OHLCV fetcher
├── indicators.py      # EMA, RSI, MACD, Bollinger, ATR, VWAP, trend_regime
│                      # Supertrend, Ichimoku, Williams %R, Donchian, PSAR
├── strategy.py        # 27 strategy classes (long-only, L/S, regime-filtered)
├── backtest.py        # Event-driven engine (2-tuple and 4-tuple signal contract)
├── portfolio.py       # PortfolioBacktest — multi-asset, 3 allocation modes
├── risk.py            # Sharpe, CAGR, max drawdown, win rate, total return
├── sizer.py           # FixedSizer, PercentEquitySizer, ATRSizer, KellySizer
├── costs.py           # NoCost, FixedCommission, PercentCommission,
│                      #   TickSlippage, PercentSlippage, CompositeCost
│                      #   + NSE presets: nse_equity_intraday / delivery
├── optimizer.py       # GridOptimizer — brute-force grid search
├── walk_forward.py    # WalkForwardValidator — anchored & rolling, warmup-aware
├── plotter.py         # plot_equity, plot_walk_forward, plot_optimizer
└── test.py            # End-to-end runner
tests/
├── conftest.py        # 7 synthetic OHLCV fixtures (no network)
├── test_indicators.py
├── test_backtest.py
├── test_risk.py
├── test_sizer.py
├── test_optimizer.py
├── test_strategies.py
├── test_costs.py
├── test_new_indicators.py
└── test_portfolio.py
```

---

## Strategy Catalogue

| Class | Type | Description |
|---|---|---|
| `EMACrossover` | Long-only | Golden/death cross |
| `EMACrossoverLS` | Long/short | Always in market, reverses on cross |
| `EMACrossoverRegime` | Regime-filtered | Long only in bull, short only in bear |
| `RSIMeanReversion` | Long-only | Buy oversold, sell overbought |
| `RSIMeanReversionLS` | Long/short | Adds short leg at overbought |
| `RSIMeanReversionRegime` | Regime-filtered | Dip-buy in bull, sell-rally in bear |
| `BollingerBreakout` | Long-only | Upper-band breakout, exit at middle |
| `BollingerBreakoutLS` | Long/short | Adds lower-band short leg |
| `BollingerBreakoutRegime` | Regime-filtered | Upper breakout in bull, breakdown in bear |
| `MACDCrossover` | Long-only | MACD/signal crossover |
| `MACDCrossoverLS` | Long/short | Always in market |
| `MACDCrossoverRegime` | Regime-filtered | Bullish cross in bull, bearish cross in bear |
| `SupertrendStrategy` | Long-only | Wilder Supertrend direction flip |
| `SupertrendLS` | Long/short | Reverses on direction flip |
| `SupertrendRegime` | Regime-filtered | Regime-gated Supertrend |
| `IchimokuStrategy` | Long-only | TK cross above the cloud |
| `IchimokuLS` | Long/short | TK cross + cloud-position filter |
| `IchimokuRegime` | Regime-filtered | Regime-gated Ichimoku |
| `WilliamsRStrategy` | Long-only | %R crosses -80 oversold threshold |
| `WilliamsRLS` | Long/short | Adds overbought short leg |
| `WilliamsRRegime` | Regime-filtered | Regime-gated Williams %R |
| `DonchianBreakout` | Long-only | Turtle Trading N-bar channel breakout |
| `DonchianBreakoutLS` | Long/short | Upper breakout long / lower breakdown short |
| `DonchianBreakoutRegime` | Regime-filtered | Regime-gated Donchian |
| `PSARStrategy` | Long-only | Parabolic SAR uptrend flip |
| `PSARLS` | Long/short | Reverses on SAR direction flip |
| `PSARRegime` | Regime-filtered | Regime-gated PSAR |

---

## Multi-Asset Portfolio

```python
from portfolio import PortfolioBacktest, EqualWeightSizer, VolTargetSizer, run_portfolio
from strategy import SupertrendRegime
from costs import nse_equity_delivery

dfs = {
    "RELIANCE.NS": rel_df,
    "INFY.NS":     inf_df,
    "HDFCBANK.NS": hdf_df,
    "TCS.NS":      tcs_df,
}

# Equal-weight, NSE delivery costs
pb = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=EqualWeightSizer(),
    cost_model=nse_equity_delivery(),
)
result = pb.run(dfs, strategy_class=SupertrendRegime)

print(result.metrics)           # portfolio-level Sharpe, CAGR, drawdown
print(result.symbol_metrics)    # per-symbol breakdown dict
result.equity_curve.plot()      # combined equity

# Inverse-volatility allocation (rebalance monthly)
pb2 = PortfolioBacktest(
    initial=1_000_000,
    portfolio_sizer=VolTargetSizer(lookback=20, rebalance_bars=21, cap=0.40),
)
result2 = pb2.run(dfs, SupertrendRegime)

# Per-symbol strategy parameters
pb3 = PortfolioBacktest(
    initial=1_000_000,
    strategy_kwargs={
        "RELIANCE.NS": {"period": 7,  "multiplier": 2.5},
        "INFY.NS":     {"period": 10, "multiplier": 3.0},
    },
)
result3 = pb3.run(dfs, SupertrendRegime)

# One-liner
result4 = run_portfolio(dfs, SupertrendRegime, initial=1_000_000)
```

### Allocation Modes

| Sizer | Logic | Best for |
|---|---|---|
| `EqualWeightSizer(cap=1.0)` | 1/N per symbol | Baseline |
| `FixedWeightSizer({sym: w})` | User-supplied static weights | Conviction-based |
| `VolTargetSizer(lookback, rebalance_bars, cap)` | Inverse-volatility, rebalanced periodically | Risk-parity |

---

## Position Sizing

```python
from sizer import FixedSizer, PercentEquitySizer, ATRSizer, KellySizer
```

| Sizer | Formula | Best for |
|---|---|---|
| `FixedSizer(shares=1)` | Always N shares | Baseline |
| `PercentEquitySizer(pct=0.10)` | `floor(equity × pct / close)` | Simple allocation |
| `ATRSizer(risk_pct=0.01, atr_mult=2.0)` | `floor(equity × risk% / (mult × ATR))` | Professional risk sizing |
| `KellySizer.from_trades(df, fraction=0.5)` | Half-Kelly f* | Theory-optimal |

Pre-attach `"atr"` column for `ATRSizer`: `df["atr"] = atr(df, 14)`.

---

## Transaction Costs

```python
from costs import (
    NoCost, FixedCommission, PercentCommission,
    TickSlippage, PercentSlippage, CompositeCost,
    nse_equity_intraday, nse_equity_delivery,
)
```

| Model | Formula | Example |
|---|---|---|
| `NoCost()` | 0 | Frictionless baseline |
| `FixedCommission(20)` | ₹20 per order leg | Zerodha intraday |
| `PercentCommission(0.0005)` | `notional × pct` | 0.05% per leg |
| `TickSlippage(1, 0.05)` | `ticks × tick_size × shares` | 1 tick NSE spread |
| `PercentSlippage(0.0002)` | `price × shares × pct` | 0.02% bid-ask |
| `CompositeCost(a, b, ...)` | `sum(each.cost(...))` | Combine any models |

### NSE presets

```python
# Zerodha intraday: ₹20 flat + 0.025% slippage
bt = Backtest(initial=100_000, cost_model=nse_equity_intraday())

# Zerodha delivery: ₹0 brokerage + 0.125% (STT + exchange + spread)
bt = Backtest(initial=100_000, cost_model=nse_equity_delivery())
```

### Custom composite

```python
cost = CompositeCost(
    FixedCommission(20),
    PercentSlippage(0.0002),
)
bt = Backtest(initial=100_000, sizer=ATRSizer(0.01), cost_model=cost)
```

### New trade record columns

When a cost model is active, `trades_df` gains two new columns:
- **`gross_pnl`** — P&L before costs
- **`cost`** — total entry + exit cost for the trade
- **`pnl`** — net P&L after costs (always present)

---

## Quick Start

```python
from indicators import atr
from strategy import EMACrossoverRegime
from backtest import Backtest
from portfolio import PortfolioBacktest, EqualWeightSizer
from sizer import ATRSizer
from costs import nse_equity_intraday
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
from plotter import plot_equity, plot_walk_forward
import yfinance as yf

df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]
df["atr"] = atr(df, period=14)

# Full realistic single-asset backtest
bt = Backtest(
    initial=100_000,
    sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0),
    cost_model=nse_equity_intraday(),
)
trades, equity, metrics = bt.run(df, EMACrossoverRegime())
print(metrics)

# Multi-asset portfolio
dfs = {sym: yf.download(sym, start="2019-01-01", end="2025-01-01") for sym in
       ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]}
for sym in dfs:
    dfs[sym].columns = [c[0].lower() for c in dfs[sym].columns]

pb = PortfolioBacktest(initial=500_000, portfolio_sizer=EqualWeightSizer())
result = pb.run(dfs, EMACrossoverRegime)
print(result.metrics)
print(result.symbol_metrics)

# Grid search
opt = GridOptimizer(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast":[5,8,12], "slow":[20,26,50], "regime_slope":[10,20]},
    min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
print(opt.best(df))

# Walk-forward
wfv = WalkForwardValidator(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast":[5,8,12], "slow":[20,26,50], "regime_slope":[10,20]},
    is_bars=378, oos_bars=252, warmup_bars=200, anchored=False,
    constraint=lambda p: p["fast"] < p["slow"],
)
result = wfv.run(df)
plot_walk_forward(result, save_path="output/wfv.png")
```

---

## Running Tests

```bash
pip install pytest pandas numpy
pytest          # runs all tests, no network required
```

---

## Roadmap

- [x] Data fetching
- [x] Core indicators: EMA, RSI, MACD, Bollinger, ATR, VWAP, trend regime
- [x] Extended indicators: Supertrend, Ichimoku, Williams %R, Donchian, PSAR
- [x] Long-only, long/short, and regime-filtered strategies (27 total)
- [x] Grid-search optimiser with constraint support
- [x] Walk-forward validation (anchored + rolling, warmup-aware)
- [x] Position sizing: Fixed, PercentEquity, ATR-based, Kelly
- [x] Results visualisation: equity curve, drawdown, per-fold OOS, optimizer scatter
- [x] pytest unit test suite (synthetic OHLCV fixtures, no network)
- [x] Commission / slippage cost model (Fixed, Percent, Tick, Composite, NSE presets)
- [x] Multi-asset portfolio backtester (Equal, Fixed, Vol-Target allocation)
- [ ] Live paper-trading hook (broker API adapter interface)
- [ ] Intraday / minute-bar support
- [ ] Regime-aware portfolio rebalancing
