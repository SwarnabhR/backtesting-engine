# backtesting-engine

A modular, from-scratch backtesting engine for systematic trading strategies.
Built in pure Python/Pandas — no `backtrader`, no `vectorbt`.

---

## Architecture

```
engine/
├── data.py            # yfinance OHLCV fetcher, normalises columns
├── indicators.py      # EMA, RSI, MACD, Bollinger, ATR, VWAP, trend_regime
├── strategy.py        # 12 strategy classes (long-only, long/short, regime-filtered)
├── backtest.py        # Event-driven engine, 2-tuple and 4-tuple signal contracts
├── risk.py            # Sharpe, CAGR, max drawdown, win rate, total return
├── sizer.py           # FixedSizer, PercentEquitySizer, ATRSizer, KellySizer
├── optimizer.py       # GridOptimizer — brute-force grid search + constraint support
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

Exits are always unfiltered — once in a trade the original exit signal applies.

---

## Position Sizing

All four sizers live in `sizer.py` and share the `PositionSizer` interface:

```python
sizer.size(equity: float, bar: pd.Series) -> int
```

| Sizer | Formula | Best for |
|---|---|---|
| `FixedSizer(shares=1)` | Always N shares | Baseline / legacy compat |
| `PercentEquitySizer(pct=0.10)` | `floor(equity × pct / close)` | Simple capital allocation |
| `ATRSizer(risk_pct=0.01, atr_mult=2.0)` | `floor(equity × risk_pct / (atr_mult × ATR))` | Professional risk-per-trade sizing |
| `KellySizer.from_trades(trades_df, fraction=0.5)` | Half-Kelly optimal fraction | Theory-optimal, high variance |

### ATRSizer setup

Pre-attach the `"atr"` column to your DataFrame before running:

```python
from indicators import atr
df["atr"] = atr(df, period=14)

bt = Backtest(initial=100_000, sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0))
```

### Backward compatibility

`position_size=N` still works and is silently converted to `FixedSizer(N)`:

```python
Backtest(initial=100_000, position_size=1)   # identical to FixedSizer(1)
```

---

## Signal Contract

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

### Default Parameters (FixedSizer, 1 share)

| Strategy | Mode | Sharpe | CAGR | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|---|
| EMA Crossover | Long-only | −0.87 | 1.69% | −2.18% | 45.5% | 11 |
| EMA Crossover | Long/short | −1.29 | 1.46% | −2.53% | 42.9% | 21 |
| EMA Crossover | **Regime-filtered** | **−0.53** | **1.61%** | **−2.18%** | **50.0%** | 6 |
| RSI Mean Reversion | Long-only | −1.23 | 0.35% | −1.74% | 71.4% | 7 |
| RSI Mean Reversion | Long/short | −2.09 | −0.80% | −6.46% | 57.1% | 14 |
| RSI Mean Reversion | **Regime-filtered** | **−1.40** | **0.23%** | **−1.13%** | **100%** | 4 |
| Bollinger Breakout | Long-only | −1.14 | 0.50% | −2.80% | 55.6% | 9 |
| Bollinger Breakout | Long/short | −2.25 | −0.61% | −4.67% | 35.0% | 20 |
| Bollinger Breakout | **Regime-filtered** | −1.86 | −0.11% | −2.85% | 37.5% | 8 |
| MACD Crossover | Long-only | −1.04 | 1.14% | −2.05% | 32.1% | 28 |
| MACD Crossover | Long/short | −1.71 | 0.35% | −5.75% | 30.9% | 55 |
| MACD Crossover | **Regime-filtered** | −1.58 | 0.26% | −2.11% | 27.3% | 22 |

### Key Findings

1. **Regime filter helps EMA and RSI, hurts Bollinger and MACD.**  
   EMA regime: Sharpe −0.87 → −0.53, DD −2.18% → −2.18%, WR 45% → 50%.  
   RSI regime: DD cut from −1.74% → −1.13%, win rate jumps to 100% (small sample).

2. **Adding short legs without regime filter always hurts on Nifty 2022–2025.**  
   RSI LS drawdown: −1.74% → −6.46%. Bollinger LS win rate collapses to 35%.

3. **All strategies underperform buy-and-hold on this bull-market dataset.**  
   Rule-based systems without ML or execution edge are expected to lag a
   strongly directional index. The value is in drawdown control, not alpha.

4. **Walk-forward OOS Sharpe (EMA anchored): −0.996 over 7 folds.**  
   IS-to-OOS Sharpe degradation is modest (≈1.0–1.2 IS vs ≈0.6–2.6 OOS per fold),
   indicating mild but real overfitting in the shorter windows.

---

## Quick Start

```python
from indicators import atr
from strategy import EMACrossoverRegime
from backtest import Backtest
from sizer import ATRSizer
from optimizer import GridOptimizer
from walk_forward import WalkForwardValidator
import yfinance as yf

df = yf.download("^NSEI", start="2019-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]
df["atr"] = atr(df, period=14)

# Single backtest with ATR sizing
bt = Backtest(initial=100_000, sizer=ATRSizer(risk_pct=0.01, atr_mult=2.0))
trades, equity, metrics = bt.run(df, EMACrossoverRegime())
print(metrics)

# Grid search
opt = GridOptimizer(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast": [5, 8, 12], "slow": [20, 26, 50], "regime_slope": [10, 20]},
    min_trades=2,
    constraint=lambda p: p["fast"] < p["slow"],
)
print(opt.best(df))

# Walk-forward (regime strategy needs warmup_bars)
wfv = WalkForwardValidator(
    strategy_class=EMACrossoverRegime,
    param_grid={"fast": [5, 8, 12], "slow": [20, 26, 50], "regime_slope": [10, 20]},
    is_bars=378, oos_bars=252, warmup_bars=200, anchored=False,
    constraint=lambda p: p["fast"] < p["slow"],
)
result = wfv.run(df)
print(result)
print(result.summary)
```

---

## Walk-Forward Validation

| Mode | IS window | Best for |
|---|---|---|
| `anchored=True` | Grows each fold (starts at bar 0) | Trend strategies |
| `anchored=False` | Fixed-length sliding window | Mean-reversion |

**`warmup_bars`** pads every IS slice so indicators with long lookbacks
(e.g. 200-bar SMA) have valid values from the first usable bar.  
The OOS window always receives the tail of the IS slice as warmup context.

---

## Roadmap

- [x] Data fetching (`data.py`)
- [x] Core indicators: EMA, RSI, MACD, Bollinger, ATR, VWAP, trend regime
- [x] Long-only strategies (EMA, RSI, Bollinger, MACD)
- [x] Long/short strategies
- [x] Regime-filtered strategies
- [x] Grid-search optimiser with constraint support
- [x] Walk-forward validation (anchored + rolling, warmup-aware)
- [x] Position sizing: FixedSizer, PercentEquity, ATR-based, Kelly
- [ ] Results visualisation: equity curve, drawdown chart, per-fold OOS plot
- [ ] `pytest` unit test suite with synthetic OHLCV fixtures
- [ ] Commission / slippage model

---

## Requirements

```
pandas
yfinance
numpy
```
