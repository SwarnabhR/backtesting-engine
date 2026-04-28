from __future__ import annotations

"""
engine/plotter.py
-----------------
Results visualisation for the backtesting engine.

Three public functions
----------------------
plot_equity(equity, trades, benchmark, title, save_path)
    Three-panel chart: equity curve + benchmark, drawdown, trade P&L.

plot_walk_forward(wf_result, title, save_path)
    Per-fold IS vs OOS Sharpe bar chart with degradation annotations.

plot_optimizer(results_df, x, y, hue, top_n, title, save_path)
    Scatter plot of the optimiser's parameter search space.

Dependencies: matplotlib (standard scientific stack, no extra install needed
beyond pandas/numpy).
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── colour palette (matches README dark-teal theme) ─────────────────────────
_C = dict(
    equity   = "#01696f",   # teal  – strategy equity
    bench    = "#aaaaaa",   # grey  – buy-and-hold benchmark
    drawdown = "#a12c7b",   # maroon
    win      = "#437a22",   # green – winning trade
    loss     = "#a13544",   # red   – losing trade
    is_bar   = "#4f98a3",   # light teal – IS Sharpe bars
    oos_bar  = "#01696f",   # dark teal  – OOS Sharpe bars
    neg_bar  = "#a12c7b",   # negative OOS
    scatter  = "#01696f",
    top      = "#da7101",   # orange – top-N results
    grid     = "#e0ddd8",
    spine    = "#bab9b4",
)

_FIGSIZE_MAIN   = (13, 9)
_FIGSIZE_WFV    = (12, 5)
_FIGSIZE_OPTIM  = (9,  6)
_DPI            = 120


# ── helpers ─────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, ylabel: str = "", xlabel: str = "") -> None:
    """Apply consistent styling to an axes."""
    ax.set_facecolor("#f9f8f5")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel(ylabel, fontsize=9, color="#7a7974")
    ax.set_xlabel(xlabel, fontsize=9, color="#7a7974")
    ax.tick_params(axis="both", labelsize=8, colors="#7a7974")
    for spine in ax.spines.values():
        spine.set_edgecolor(_C["spine"])
        spine.set_linewidth(0.6)
    ax.grid(axis="y", color=_C["grid"], linewidth=0.5, linestyle="--")
    ax.grid(axis="x", color=_C["grid"], linewidth=0.4, linestyle=":")


def _save_or_show(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        print(f"[plotter] saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _benchmark_equity(df_close: pd.Series, initial: float) -> pd.Series:
    """Buy-and-hold equity normalised to *initial*."""
    return initial * (df_close / df_close.iloc[0])


# ── 1. Main equity / drawdown / trade chart ──────────────────────────────────

def plot_equity(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    title: str = "Backtest Results",
    save_path: Optional[str | Path] = None,
    initial: float = 100_000,
) -> None:
    """
    Three-panel figure:
      Panel 1 (tall)  – strategy equity curve + optional benchmark.
      Panel 2 (medium)– drawdown as a filled area chart.
      Panel 3 (short) – per-trade P&L as a stem plot (win = green, loss = red).

    Parameters
    ----------
    equity      : pd.Series  – equity curve returned by Backtest.run()
    trades      : pd.DataFrame or None – trades_df returned by Backtest.run()
    benchmark   : pd.Series or None – buy-and-hold close prices (same index).
                  If provided, aligned and normalised to *initial*.
    title       : figure title.
    save_path   : file path to save the PNG (e.g. "output/ema_result.png").
                  None = show interactively (requires a display).
    initial     : starting capital for benchmark normalisation.
    """
    has_trades  = trades is not None and not trades.empty
    n_panels    = 3 if has_trades else 2
    heights     = [4, 1.5, 1] if has_trades else [4, 1.5]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=_FIGSIZE_MAIN,
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    fig.patch.set_facecolor("#f7f6f2")

    ax_eq = axes[0]
    ax_dd = axes[1]
    ax_tr = axes[2] if has_trades else None

    # ── Panel 1: equity ──────────────────────────────────────────────────────
    _style_ax(ax_eq, ylabel="Portfolio Value (₹)")
    ax_eq.plot(equity.index, equity.values,
               color=_C["equity"], linewidth=1.5, label="Strategy", zorder=3)

    if benchmark is not None:
        bench_aligned = benchmark.reindex(equity.index, method="ffill")
        bench_norm    = initial * (bench_aligned / bench_aligned.iloc[0])
        ax_eq.plot(bench_norm.index, bench_norm.values,
                   color=_C["bench"], linewidth=1.0, linestyle="--",
                   label="Buy & Hold", zorder=2)

    ax_eq.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
    )

    # shade region above/below initial capital
    ax_eq.axhline(initial, color=_C["spine"], linewidth=0.7, linestyle=":")
    ax_eq.fill_between(
        equity.index, equity.values, initial,
        where=(equity.values >= initial), alpha=0.08, color=_C["equity"]
    )
    ax_eq.fill_between(
        equity.index, equity.values, initial,
        where=(equity.values < initial), alpha=0.08, color=_C["drawdown"]
    )

    # trade entry/exit markers
    if has_trades:
        for _, row in trades.iterrows():
            color = _C["win"] if row["pnl"] >= 0 else _C["loss"]
            ax_eq.axvline(row["entry_date"], color=color,
                          linewidth=0.4, alpha=0.35)

    # metrics annotation box
    final_val = equity.iloc[-1]
    total_ret = (final_val - initial) / initial
    max_dd    = float(_drawdown_series(equity).min())
    from risk import sharpe_ratio, cagr
    sr  = sharpe_ratio(equity)
    cgr = cagr(equity)
    wr  = (
        (trades["pnl"] > 0).sum() / len(trades) if has_trades else float("nan")
    )
    n_trades = len(trades) if has_trades else 0

    ann = (
        f"Sharpe {sr:+.2f}  |  CAGR {cgr:.2%}  |  "
        f"Max DD {max_dd:.2%}  |  "
        f"Win Rate {wr:.1%}  |  Trades {n_trades}  |  "
        f"Total Return {total_ret:+.2%}"
    )
    ax_eq.set_title(ann, fontsize=8.5, color="#7a7974",
                    pad=4, loc="left")
    ax_eq.text(
        0.01, 0.97, title,
        transform=ax_eq.transAxes,
        fontsize=13, fontweight="bold", color="#28251d",
        va="top", ha="left",
    )

    # legend
    handles = [mpatches.Patch(color=_C["equity"], label="Strategy")]
    if benchmark is not None:
        handles.append(mpatches.Patch(color=_C["bench"], label="Buy & Hold"))
    ax_eq.legend(handles=handles, loc="upper left", fontsize=8,
                 framealpha=0.6, edgecolor=_C["spine"])

    # ── Panel 2: drawdown ────────────────────────────────────────────────────
    _style_ax(ax_dd, ylabel="Drawdown")
    dd = _drawdown_series(equity)
    ax_dd.fill_between(dd.index, dd.values, 0,
                        color=_C["drawdown"], alpha=0.55)
    ax_dd.plot(dd.index, dd.values,
               color=_C["drawdown"], linewidth=0.8)
    ax_dd.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1%}")
    )
    ax_dd.axhline(0, color=_C["spine"], linewidth=0.6)

    # ── Panel 3: per-trade P&L ───────────────────────────────────────────────
    if has_trades and ax_tr is not None:
        _style_ax(ax_tr, ylabel="Trade P&L (₹)", xlabel="Date")
        ax_tr.axhline(0, color=_C["spine"], linewidth=0.6)
        for _, row in trades.iterrows():
            color  = _C["win"] if row["pnl"] >= 0 else _C["loss"]
            date   = row["exit_date"]
            pnl    = row["pnl"]
            ax_tr.vlines(date, 0, pnl, color=color, linewidth=1.2, alpha=0.8)
            ax_tr.scatter([date], [pnl], color=color, s=12, zorder=4)
        ax_tr.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
        )

    fig.subplots_adjust(hspace=0.05)
    _save_or_show(fig, save_path)


# ── 2. Walk-forward IS vs OOS Sharpe ─────────────────────────────────────────

def plot_walk_forward(
    wf_result,
    title: str = "Walk-Forward Validation",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Grouped bar chart: for each WFV fold, show IS Sharpe (light teal) vs
    OOS Sharpe (dark teal / maroon if negative).  Annotates the
    IS→OOS degradation (Δ) above each pair.

    Parameters
    ----------
    wf_result : WalkForwardResult
        Object returned by WalkForwardValidator.run().
    title     : figure title.
    save_path : PNG path or None.
    """
    summary = wf_result.summary.copy()
    if summary.empty:
        warnings.warn("[plotter] walk-forward summary is empty — nothing to plot.")
        return

    n_folds   = len(summary)
    x         = np.arange(n_folds)
    width     = 0.35

    fig, ax = plt.subplots(figsize=_FIGSIZE_WFV)
    fig.patch.set_facecolor("#f7f6f2")
    _style_ax(ax, ylabel="Sharpe Ratio", xlabel="Fold")

    is_sharpe  = summary["is_sharpe"].astype(float)
    oos_sharpe = summary["oos_sharpe"].astype(float)

    # IS bars (always teal)
    ax.bar(x - width / 2, is_sharpe, width,
           color=_C["is_bar"], alpha=0.80, label="IS Sharpe", zorder=3)

    # OOS bars (dark teal if ≥0, maroon if <0)
    oos_colors = [
        _C["oos_bar"] if v >= 0 else _C["neg_bar"]
        for v in oos_sharpe
    ]
    ax.bar(x + width / 2, oos_sharpe, width,
           color=oos_colors, alpha=0.90, label="OOS Sharpe", zorder=3)

    # degradation annotations
    for i, (is_v, oos_v) in enumerate(zip(is_sharpe, oos_sharpe)):
        delta = oos_v - is_v
        top   = max(is_v, oos_v, 0) + 0.05
        color = _C["win"] if delta >= 0 else _C["loss"]
        ax.text(x[i], top, f"Δ{delta:+.2f}",
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold")

    # best-params annotation below each fold
    if "best_params" in summary.columns:
        for i, params in enumerate(summary["best_params"]):
            param_str = str(params)[:30]  # truncate long param dicts
            ax.text(x[i], ax.get_ylim()[0] - 0.05, param_str,
                    ha="center", va="top", fontsize=6.0,
                    color="#7a7974", rotation=0)

    ax.axhline(0, color=_C["spine"], linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)], fontsize=8)

    # summary stats in title
    mean_oos = float(oos_sharpe.mean())
    pct_pos  = (oos_sharpe >= 0).mean() * 100
    ax.set_title(
        f"{title}\n"
        f"Mean OOS Sharpe: {mean_oos:+.3f}  |  "
        f"Positive OOS folds: {pct_pos:.0f}%  |  "
        f"{n_folds} folds",
        fontsize=10, color="#28251d", pad=8,
    )

    handles = [
        mpatches.Patch(color=_C["is_bar"],  label="IS Sharpe"),
        mpatches.Patch(color=_C["oos_bar"], label="OOS Sharpe (≥0)"),
        mpatches.Patch(color=_C["neg_bar"], label="OOS Sharpe (<0)"),
    ]
    ax.legend(handles=handles, fontsize=8,
              framealpha=0.6, edgecolor=_C["spine"])

    _save_or_show(fig, save_path)


# ── 3. Optimiser parameter scatter ──────────────────────────────────────────

def plot_optimizer(
    results_df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    color_metric: str = "sharpe_ratio",
    top_n: int = 10,
    title: str = "Optimiser Results",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Scatter plot of the optimiser's full parameter search space.

    Axes    : two chosen parameter columns (*x*, *y*).
    Colour  : mapped to *color_metric* (default: sharpe_ratio) via a
              sequential colourmap — darker = better.
    Markers : top-N results are highlighted with a larger orange ring.

    Parameters
    ----------
    results_df   : pd.DataFrame returned by GridOptimizer.run().
    x, y         : column names from results_df to use as axes
                   (typically parameter names like ``"fast"``, ``"slow"``).
    hue          : optional third parameter to split into sub-plots
                   (one axes column per unique value).  None = single plot.
    color_metric : metric column to encode as colour intensity.
    top_n        : number of top results to highlight.
    title        : figure title.
    save_path    : PNG path or None.
    """
    if results_df.empty:
        warnings.warn("[plotter] results_df is empty — nothing to plot.")
        return

    df = results_df.copy()
    if color_metric not in df.columns:
        warnings.warn(f"[plotter] '{color_metric}' not in results_df columns.")
        color_metric = df.columns[-1]

    hue_vals = sorted(df[hue].unique()) if hue and hue in df.columns else [None]
    n_cols   = len(hue_vals)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(7, 5 * n_cols), 5),
        squeeze=False,
    )
    fig.patch.set_facecolor("#f7f6f2")

    metric_min = df[color_metric].min()
    metric_max = df[color_metric].max()
    cmap       = plt.cm.YlGn  # sequential: light=low, dark=high

    for col_i, hue_val in enumerate(hue_vals):
        ax  = axes[0][col_i]
        sub = df if hue_val is None else df[df[hue] == hue_val]
        _style_ax(ax, ylabel=y if col_i == 0 else "", xlabel=x)

        # normalise metric to [0,1] for colour mapping
        norm = matplotlib.colors.Normalize(
            vmin=metric_min, vmax=metric_max
        )
        colours = [cmap(norm(v)) for v in sub[color_metric]]

        ax.scatter(
            sub[x], sub[y], c=colours,
            s=55, zorder=3, edgecolors="none", alpha=0.85,
        )

        # top-N highlight ring
        top_sub = sub.nlargest(top_n, color_metric)
        ax.scatter(
            top_sub[x], top_sub[y],
            s=130, zorder=4, facecolors="none",
            edgecolors=_C["top"], linewidths=1.5,
            label=f"Top {top_n}",
        )

        sub_title = f"{hue}={hue_val}" if hue_val is not None else title
        ax.set_title(sub_title, fontsize=9, color="#28251d")
        ax.legend(fontsize=7.5, framealpha=0.5, edgecolor=_C["spine"])

    # colourbar for metric
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(
        vmin=metric_min, vmax=metric_max
    ))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], shrink=0.85, pad=0.02)
    cbar.set_label(color_metric, fontsize=8, color="#7a7974")
    cbar.ax.tick_params(labelsize=7)

    if len(hue_vals) == 1:
        axes[0][0].set_title(title, fontsize=10, color="#28251d")

    _save_or_show(fig, save_path)
