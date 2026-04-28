from __future__ import annotations

import pandas as pd
from risk import compute_metrics, BacktestMetrics
from sizer import PositionSizer, FixedSizer
from costs import CostModel, NoCost


class Backtest:
    """
    Event-driven backtester supporting long-only and long/short strategies.

    Signal contract
    ---------------
    Strategies may return either:
      - A 2-tuple ``(entries, exits)``             -> long-only
      - A 4-tuple ``(long_entries, long_exits,
                     short_entries, short_exits)``  -> long/short

    Position sizing
    ---------------
    Pass any :class:`sizer.PositionSizer` subclass as *sizer*.
    Defaults to :class:`sizer.FixedSizer` (1 share) for backward compat.
    The legacy ``position_size: int`` kwarg is still accepted and is
    converted to ``FixedSizer(position_size)`` automatically.

    Transaction costs
    -----------------
    Pass any :class:`costs.CostModel` subclass as *cost_model*.
    Defaults to :class:`costs.NoCost` (zero friction) for backward compat.
    The engine calls ``cost_model.cost(price, shares)`` at both entry and
    exit; the sum is subtracted from the trade P&L and accumulated into
    the equity curve.
    """

    def __init__(
        self,
        initial: float = 10_000,
        position_size: int = 1,
        sizer: PositionSizer | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.initial     = initial
        self.sizer       = sizer if sizer is not None else FixedSizer(position_size)
        self.cost_model  = cost_model if cost_model is not None else NoCost()

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
    # Long-only
    # ------------------------------------------------------------------

    def _run_long_only(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        position    = 0
        entry_date  = None
        entry_price = None
        entry_cost  = 0.0
        shares_held = 0
        trades: list[dict] = []
        equity_values: list[float] = []
        realised_pnl = 0.0

        for i in range(1, len(df)):
            close = df["close"].iloc[i]
            date  = df.index[i]
            bar   = df.iloc[i]
            current_equity = self.initial + realised_pnl

            if entries.iloc[i] and position == 0:
                shares_held = self.sizer.size(current_equity, bar)
                entry_cost  = self.cost_model.cost(close, shares_held)
                position    = 1
                entry_date  = date
                entry_price = close

            elif exits.iloc[i] and position == 1:
                exit_cost    = self.cost_model.cost(close, shares_held)
                gross_pnl    = (close - entry_price) * shares_held
                total_cost   = entry_cost + exit_cost
                pnl          = gross_pnl - total_cost
                realised_pnl += pnl
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close,
                        "long", gross_pnl, pnl, total_cost, shares_held
                    )
                )
                position   = 0
                entry_cost = 0.0

            unrealized = (close - entry_price) * shares_held if position == 1 else 0.0
            # unrealized equity does not subtract open entry cost (not yet paid)
            equity_values.append(self.initial + realised_pnl + unrealized)

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
        position    = 0
        entry_date  = None
        entry_price = None
        entry_cost  = 0.0
        shares_held = 0
        trades: list[dict] = []
        equity_values: list[float] = []
        realised_pnl = 0.0

        for i in range(1, len(df)):
            close = df["close"].iloc[i]
            date  = df.index[i]
            bar   = df.iloc[i]
            current_equity = self.initial + realised_pnl

            # --- exit first ---
            if position == 1 and long_exits.iloc[i]:
                exit_cost    = self.cost_model.cost(close, shares_held)
                gross_pnl    = (close - entry_price) * shares_held
                total_cost   = entry_cost + exit_cost
                pnl          = gross_pnl - total_cost
                realised_pnl += pnl
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close,
                        "long", gross_pnl, pnl, total_cost, shares_held
                    )
                )
                position   = 0
                entry_cost = 0.0

            elif position == -1 and short_exits.iloc[i]:
                exit_cost    = self.cost_model.cost(close, shares_held)
                gross_pnl    = (entry_price - close) * shares_held
                total_cost   = entry_cost + exit_cost
                pnl          = gross_pnl - total_cost
                realised_pnl += pnl
                trades.append(
                    self._trade_record(
                        entry_date, date, entry_price, close,
                        "short", gross_pnl, pnl, total_cost, shares_held
                    )
                )
                position   = 0
                entry_cost = 0.0

            # --- enter ---
            if position == 0:
                if long_entries.iloc[i]:
                    shares_held = self.sizer.size(current_equity, bar)
                    entry_cost  = self.cost_model.cost(close, shares_held)
                    position    = 1
                    entry_date  = date
                    entry_price = close
                elif short_entries.iloc[i]:
                    shares_held = self.sizer.size(current_equity, bar)
                    entry_cost  = self.cost_model.cost(close, shares_held)
                    position    = -1
                    entry_date  = date
                    entry_price = close

            if position == 1:
                unrealized = (close - entry_price) * shares_held
            elif position == -1:
                unrealized = (entry_price - close) * shares_held
            else:
                unrealized = 0.0

            equity_values.append(self.initial + realised_pnl + unrealized)

        return self._finalise(trades, equity_values, df)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_record(
        entry_date, exit_date,
        entry_price: float, exit_price: float,
        direction: str,
        gross_pnl: float, net_pnl: float, total_cost: float,
        shares: int,
    ) -> dict:
        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        return {
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "direction":   direction,
            "shares":      shares,
            "gross_pnl":   gross_pnl,
            "cost":        total_cost,
            "pnl":         net_pnl,
            "pnl_pct":     pnl_pct,
        }

    def _finalise(
        self,
        trades: list[dict],
        equity_values: list[float],
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, BacktestMetrics]:
        trades_df    = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values, index=df.index[1:])
        metrics      = compute_metrics(trades_df, equity_curve)
        return trades_df, equity_curve, metrics
