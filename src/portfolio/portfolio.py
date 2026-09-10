"""
Portfolio Layer & Compounded Risk Metrics.
Computes multi-year compounded returns, drawdowns, Sharpe, Sortino, Calmar, and monthly return matrices.
"""

from dataclasses import dataclass, field
import datetime as dt
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    initial_equity: float
    final_equity: float
    total_net_pnl: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_usd: float
    monthly_returns_matrix: Dict[int, Dict[int, float]]  # Year -> Month -> Pct


class PortfolioEngine:
    def __init__(self, initial_equity: float = 50000.0):
        self.initial_equity = initial_equity

    def compute_metrics(
        self,
        daily_equity_series: pd.Series,  # index is dt.date, values are account equity
        trade_pnls: List[float],
    ) -> PortfolioMetrics:
        """Computes comprehensive portfolio performance analytics."""
        if daily_equity_series.empty or len(daily_equity_series) < 2:
            return PortfolioMetrics(
                initial_equity=self.initial_equity,
                final_equity=self.initial_equity,
                total_net_pnl=0.0,
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_usd=0.0,
                total_trades=len(trade_pnls),
                winning_trades=sum(1 for p in trade_pnls if p > 0),
                losing_trades=sum(1 for p in trade_pnls if p < 0),
                win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_usd=0.0,
                monthly_returns_matrix={},
            )

        final_equity = float(daily_equity_series.iloc[-1])
        total_pnl = final_equity - self.initial_equity
        total_ret_pct = (total_pnl / self.initial_equity) * 100.0

        # Daily pct returns
        daily_returns = daily_equity_series.pct_change().dropna()
        n_days = len(daily_returns)

        # Annualized return
        ann_return_pct = ((1.0 + total_ret_pct / 100.0) ** (252.0 / max(n_days, 1)) - 1.0) * 100.0

        # Sharpe ratio
        mean_ret = float(daily_returns.mean())
        std_ret = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 1e-6
        sharpe = (mean_ret / std_ret) * np.sqrt(252.0) if std_ret > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = float(np.sqrt(np.mean(downside_returns ** 2))) if len(downside_returns) > 0 else 1e-6
        sortino = (mean_ret / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0

        # Max drawdown
        peaks = daily_equity_series.cummax()
        drawdown_series = (daily_equity_series - peaks) / peaks
        max_dd_pct = abs(float(drawdown_series.min())) * 100.0
        max_dd_usd = abs(float((daily_equity_series - peaks).min()))

        # Calmar ratio
        calmar = (ann_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Trade stats
        total_trades = len(trade_pnls)
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        expectancy = float(np.mean(trade_pnls)) if trade_pnls else 0.0

        # Monthly returns matrix
        monthly_matrix: Dict[int, Dict[int, float]] = {}
        # Convert daily series index to datetime if not already
        daily_dt_index = pd.to_datetime(daily_equity_series.index)
        temp_df = pd.DataFrame({"equity": daily_equity_series.values}, index=daily_dt_index)
        monthly_groups = temp_df.resample("ME").last().pct_change().dropna()

        for timestamp, row in monthly_groups.iterrows():
            yr = timestamp.year
            mo = timestamp.month
            if yr not in monthly_matrix:
                monthly_matrix[yr] = {}
            monthly_matrix[yr][mo] = round(float(row["equity"]) * 100.0, 2)

        return PortfolioMetrics(
            initial_equity=self.initial_equity,
            final_equity=round(final_equity, 2),
            total_net_pnl=round(total_pnl, 2),
            total_return_pct=round(total_ret_pct, 2),
            annualized_return_pct=round(ann_return_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_usd=round(max_dd_usd, 2),
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            expectancy_usd=round(expectancy, 2),
            monthly_returns_matrix=monthly_matrix,
        )
