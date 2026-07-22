"""Typed domain models used by the quantitative pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PortfolioPosition:
    """A selected portfolio position."""

    ticker: str
    weight: float
    price: float | None
    score: float


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary statistics for a backtest return series."""

    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    volatility: float
    cumulative_return: float
    alpha: float
    beta: float
    tracking_error: float


@dataclass(frozen=True)
class RebalanceEvent:
    """Rebalance metadata."""

    date: date
    tickers: tuple[str, ...]
