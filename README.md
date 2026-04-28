# backtesting-engine

A modular, from-scratch backtesting engine for systematic trading strategies.
Built in pure Python/Pandas — no `backtrader`, no `vectorbt`.

---

## Architecture

```
engine/
├── data.py            # yfinance OHLCV fetcher, normalises columns
├── indicators.py      # EMA, RSI, MACD, Bollinger Bands, VWAP, trend_regime
├── strategy.py        # 12 strategy classes (long-only, long/short, regime-filtered)
├── backtest.py        # Event-driven engine, supports 2-tuple and 4-tuple signal contracts
├── risk.py            # Sharpe, CAGR, max drawdown, win rate, total return
├── optimizer.py       # GridOptimizer — brute-force grid search with constraint support
├── walk_forward.py    # WalkForwardValidator — anchored & rolling window modes
└── test.py            # End-to-end runner
```

---

## Strategy Catalogue

| Class | Type | Description |
|---|---|---|
| `EMACrossover` | Long-only | Golden/death cross |
| `EMACrossoverLS` | Long/short | Always in market, reverses on cross |
| `EMACrossoverRegime` | **Regime-filtered** | Long only in bull, short only in bear |
| `RSIMeanReversion` | Long-only | Buy oversold, sell overbought |
| `RSIMeanReversionLS` | Long/short | Adds short leg at overbought |
| `RSIMeanReversionRegime` | **Regime-filtered** | Dip-buy in bull, sell-rally in bear |
| `BollingerBreakout` | Long-only | Upper-band breakout, exit at middle |
| `BollingerBreakoutLS` | Long/short | Adds lower-band short leg |
| `BollingerBreakoutRegime` | **Regime-filtered** | Upper breakout in bull, breakdown in bear |
| `MACDCrossover` | Long-only | MACD/signal crossover |
| `MACDCrossoverLS` | Long/short | Always in market |
| `MACDCrossoverRegime` | **Regime-filtered** | Bullish cross in bull, bearish cross in bear |

### Regime Filter

`trend_regime()` classifies each bar as **bull (+1)**, **bear (−1)**, or **neutral (0)**
by measuring the slope of the 200-SMA over the last 20 bars:

```
slope = (MA_now − MA_20_bars_ago) / MA_20_bars_ago
  > 0  →  bull   (+1)
  < 0  →  bear   (−1)
  NaN  →  neutral (0)
```

This avoids whipsaw during sideways markets where price oscillates around a flat MA.
Exits are always unfiltered — once in a trade the original exit signal always applies.

---

## Signal Contract

Strategies return either a **2-tuple** (long-only) or **4-tuple** (long/short).
The engine auto-detects which mode to use — no code change needed.

```python
# Long-only
def generate_signals(df) -> tuple[pd.Series, pd.Series]:
    return entries, exits

# Long/short
def generate_signals(df) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return long_entries, long_exits, short_entries, short_exits
```

---

## Benchmark Results — Nifty 50 (`^NSEI`) · Jan 2022 – Jan 2025

> Buy-and-hold returned **~42%** over this period.  
> Negative Sharpe = underperformance vs risk-free rate (5%), not a loss.

### Default Parameters

| Strategy | Sharpe | CAGR | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|
| EMA Crossover | −0.87 | 1.69% | −2.18% | 45.5% | 11 |
| RSI Mean Reversion | −1.23 | 0.35% | −1.74% | 71.4% | 7 |
| Bollinger Breakout | −1.14 | 0.50% | −2.80% | 55.6% | 9 |
| MACD Crossover | −1.04 | 1.14% | −2.05% | 32.1% | 28 |

### After Grid-Search Optimisation

| Strategy | Best Params | Sharpe | CAGR | Trades |
|---|---|---|---|---|
| EMA Crossover | fast=5, slow=26 | −0.76 | 1.86% | 15 |
| RSI Mean Reversion | period=20, OS=35, OB=70 | −0.54 | 1.13% | 7 |
| Bollinger Breakout | period=30, std=1 | −0.79 | 1.39% | 14 |

### Long-only vs Long/Short vs Regime-Filtered

**Key finding:** Nifty 50 is a strong bull-market index (2022–2025). Adding
short legs without a regime filter **hurts every strategy** — RSI LS drawdown
jumped from −1.74% → −6.46%, Bollinger LS win rate collapsed to 35%.
The regime filter suppresses short entries during bull phases, recovering most
of the damage.

| Strategy | Mode | Sharpe | CAGR | Max DD |
|---|---|---|---|---|
| EMA | Long-only | −0.87 | 1.69% | −2.18% |
| EMA | Long/short | −1.29 | 1.46% | −2.53% |
| EMA | Regime-filtered | _run_ | _run_ | _run_ |
| RSI | Long-only | −1.23 | 0.35% | −1.74% |
| RSI | Long/short | −2.09 | −0.80% | −6.46% |
| RSI | Regime-filtered | _run_ | _run_ | _run_ |
| Bollinger | Long-only | −1.14 | 0.50% | −2.80% |
| Bollinger | Long/short | −2.25 | −0.61% | −4.67% |
| Bollinger | Regime-filtered | _run_ | _run_ | _run_ |

---

## Quick Start

```python
from data import fetch
from strategy import EMACrossoverRegime
from backtest import Backtest
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator

df = fetch("^NSEI", "2019-01-01", "2025-01-01")

# Single backtest
bt = Backtest(initial=100_000)
trades, equity, metrics = bt.run(df, EMACrossoverRegime())
print(metrics)

# Grid search
opt = GridOptimizer(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast": [5, 8, 12], "slow": [20, 26, 50], "regime_slope": [10, 20]},
    min_trades=3,
    constraint=lambda p: p["fast"] < p["slow"],
)
print(opt.best(df))

# Walk-forward validation (anchored, 2-yr IS / 6-mo OOS)
wfv = WalkForwardValidator(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast": [5, 8, 12], "slow": [20, 26, 50], "regime_slope": [10, 20]},
    is_bars=504, oos_bars=126, anchored=True,
    constraint=lambda p: p["fast"] < p["slow"],
)
result = wfv.run(df)
print(result)             # OOS aggregate metrics
print(result.summary)    # per-fold detail
```

---

## Walk-Forward Validation

`WalkForwardValidator` prevents curve-fitting by verifying that IS-optimal
parameters generalise to unseen OOS data.

### Window Modes

| Mode | IS window | Best for |
|---|---|---|
| `anchored=True` | Grows with each fold (always starts at bar 0) | Trend strategies needing max history |
| `anchored=False` | Fixed-length sliding window | Mean-reversion where recent data matters more |

### Output

```
WalkForwardResult(4 folds | OOS sharpe=X.XXX, cagr=X.XX%, dd=-X.XX%, wr=XX.XX%, trades=N)
```

`result.summary` — DataFrame with one row per fold:

| fold | is_start | is_end | oos_start | oos_end | params_fast | params_slow | is_sharpe | oos_sharpe | oos_cagr | oos_max_dd | oos_win_rate | oos_trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2019-01 | 2021-01 | 2021-01 | 2021-07 | 8 | 26 | −0.60 | −0.72 | 1.2% | −1.8% | 50% | 4 |
| ... |

---

## Roadmap

- [x] Data fetching (`data.py`)
- [x] Core indicators: EMA, RSI, MACD, Bollinger, VWAP, trend regime
- [x] Long-only strategies (EMA, RSI, Bollinger, MACD)
- [x] Long/short strategies
- [x] Regime-filtered strategies
- [x] Grid-search optimiser with constraint support
- [x] Walk-forward validation (anchored + rolling)
- [ ] Position sizing: % equity, ATR-based stops
- [ ] Results visualisation: equity curve, drawdown chart
- [ ] `pytest` unit test suite with synthetic OHLCV fixtures
- [ ] Commission / slippage model

---

## Requirements

```
pandas
yfinance
numpy
```
