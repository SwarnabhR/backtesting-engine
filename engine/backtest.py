from __future__ import annotations

import pandas as pd
from risk import compute_metrics, BacktestMetrics


class Backtest:
    """
    Event-driven backtester supporting long-only and long/short strategies.

    Signal contract
    ---------------
    Strategies may return either:
      - A 2-tuple ``(entries, exits)``          → long-only (original behaviour)
      - A 4-tuple ``(long_entries, long_exits,
                     short_entries, short_exits)`` → long/short mode

    Both ``entries``/``exits`` variants are boolean ``pd.Series`` aligned
    to *df*'s index.

    Position encoding
    -----------------
      0  = flat
      1  = long
     -1  = short
    """

    def __init__(self, initial: float = 10_000, position_size: int = 1) -> None:
        self.initial = initial
        self.position_size = position_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, df: pd.DataFrame, strategy
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        signals = strategy.generate_signals(df)

        if len(signals) == 4:
            return self._run_long_short(df, *signals)
        elif len(signals) == 2:
            return self._run_long_only(df, *signals)
        else:
            raise ValueError(
                f"generate_signals() must return a 2- or 4-tuple, got {len(signals)}"
            )

    # ------------------------------------------------------------------
    # Long-only (original behaviour, unchanged)
    # ------------------------------------------------------------------

    def _run_long_only(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        position = 0
        entry_date = None
        entry_price = None
        trades: list[dict] = []
        equity_values: list[float] = []

        for i in range(1, len(df)):
            close = df["close"].iloc[i]
            date = df.index[i]

            if entries.iloc[i] and position == 0:
                position = 1
                entry_date = date
                entry_price = close

            elif exits.iloc[i] and position == 1:
                pnl = (close - entry_price) * self.position_size
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close, "long", pnl
                    )
                )
                position = 0

            unrealized = (
                (close - entry_price) * self.position_size if position == 1 else 0.0
            )
            equity = self.initial + sum(t["pnl"] for t in trades) + unrealized
            equity_values.append(equity)

        return self._finalise(trades, equity_values, df)

    # ------------------------------------------------------------------
    # Long / Short
    # ------------------------------------------------------------------

    def _run_long_short(
        self,
        df: pd.DataFrame,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        position = 0       # 0=flat, 1=long, -1=short
        entry_date = None
        entry_price = None
        trades: list[dict] = []
        equity_values: list[float] = []

        for i in range(1, len(df)):
            close = df["close"].iloc[i]
            date = df.index[i]

            # --- exit any open position first ---
            if position == 1 and long_exits.iloc[i]:
                pnl = (close - entry_price) * self.position_size
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close, "long", pnl
                    )
                )
                position = 0

            elif position == -1 and short_exits.iloc[i]:
                pnl = (entry_price - close) * self.position_size
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close, "short", pnl
                    )
                )
                position = 0

            # --- enter new position (only when flat) ---
            if position == 0:
                if long_entries.iloc[i]:
                    position = 1
                    entry_date = date
                    entry_price = close
                elif short_entries.iloc[i]:
                    position = -1
                    entry_date = date
                    entry_price = close

            # unrealized P&L for equity curve
            if position == 1:
                unrealized = (close - entry_price) * self.position_size
            elif position == -1:
                unrealized = (entry_price - close) * self.position_size
            else:
                unrealized = 0.0

            equity = self.initial + sum(t["pnl"] for t in trades) + unrealized
            equity_values.append(equity)

        return self._finalise(trades, equity_values, df)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_record(
        entry_date, exit_date, entry_price: float, exit_price: float,
        direction: str, pnl: float
    ) -> dict:
        pnl_pct = (
            (exit_price - entry_price) / entry_price
            if direction == "long"
            else (entry_price - exit_price) / entry_price
        )
        return {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "shares": 0,          # kept for schema compatibility
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    def _finalise(
        self,
        trades: list[dict],
        equity_values: list[float],
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values, index=df.index[1:])
        metrics = compute_metrics(trades_df, equity_curve)
        return trades_df, equity_curve, metrics
