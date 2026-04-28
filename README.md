# backtesting-engine

A modular, from-scratch backtesting engine for systematic trading strategies.
Built in pure Python/Pandas — no `backtrader`, no `vectorbt`.

---

## Architecture

```
engine/
├── data.py            # yfinance OHLCV fetcher
├── indicators.py      # EMA, RSI, MACD, Bollinger, ATR, VWAP, trend_regime
├── strategy.py        # 12 strategy classes (long-only, L/S, regime-filtered)
├── backtest.py        # Event-driven engine (2-tuple and 4-tuple signal contract)
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
└── test_costs.py
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
from sizer import ATRSizer
from costs import nse_equity_intraday
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
from plotter import plot_equity, plot_walk_forward
import yfinance as yf

df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]
df["atr"] = atr(df, period=14)

# Full realistic backtest
bt = Backtest(
    initial=100_000,
    sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0),
    cost_model=nse_equity_intraday(),
)
trades, equity, metrics = bt.run(df, EMACrossoverRegime())
print(metrics)
print(trades[["entry_date","exit_date","shares","gross_pnl","cost","pnl"]].tail())

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
pytest          # runs all ~95 tests, no network required
```

---

## Roadmap

- [x] Data fetching
- [x] Core indicators: EMA, RSI, MACD, Bollinger, ATR, VWAP, trend regime
- [x] Long-only, long/short, and regime-filtered strategies (12 total)
- [x] Grid-search optimiser with constraint support
- [x] Walk-forward validation (anchored + rolling, warmup-aware)
- [x] Position sizing: Fixed, PercentEquity, ATR-based, Kelly
- [x] Results visualisation: equity curve, drawdown, per-fold OOS, optimizer scatter
- [x] pytest unit test suite (~95 tests, synthetic OHLCV fixtures)
- [x] Commission / slippage cost model (Fixed, Percent, Tick, Composite, NSE presets)

---

## Requirements

```
pandas
numpy
yfinance
matplotlib
pytest
```
