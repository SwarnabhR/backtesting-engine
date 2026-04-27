from __future__ import annotations

import pandas as pd 
from risk import computer_metrics, BacktestMetrics

class Backtest:
    def __init__(self, initial: float=10000, position_size: int=1) -> None:
        self.initial = initial
        self.position_size = position_size
        
    def run(self, df: pd.DataFrame, strategy) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        entries, exits = strategy.generate_signals(df)
        
        position = 0
        entry_date = None
        entry_price = None
        trades: list[dict] = []
        equity_values: list[dict] = []
        
        for i in range(1, len(df)):
            close = df['close'].iloc[i]
            date = df.index[i]
            
            if entries.iloc[i] and position == 0:
                position = 1
                entry_date = date
                entry_price = close
            
            elif exits.iloc[i] and position == 1:
                pnl = (close - entry_price) * self.position_size
                pnl_pct = (close - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'shares': self.position_size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                position = 0
            unrealized = (close - entry_price) * self.position_size if position == 1 else 0.0
            equity = self.initial + sum(t['pnl'] for t in trades) + unrealized
            equity_values.append(equity)
        
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values, index=df.index[1:])
        
        metrics = computer_metrics(trades_df, equity_curve)
        
        return trades_df, equity_curve, metrics