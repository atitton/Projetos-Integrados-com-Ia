"""Vectorized backtesting utilities for quarterly value portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    """Convert periodic returns into cumulative wealth."""
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(curve: pd.Series) -> float:
    """Return maximum drawdown from an equity curve."""
    drawdown = curve / curve.cummax() - 1
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, benchmark: pd.Series | None = None, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Calculate CAGR, Sharpe, Sortino, drawdown, alpha, beta and tracking error."""
    returns = returns.dropna()
    if returns.empty:
        return {key: 0.0 for key in ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "cumulative_return", "alpha", "beta", "tracking_error"]}
    curve = equity_curve(returns)
    years = max(len(returns) / 252, 1 / 252)
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(252)
    excess = returns.mean() * 252 - risk_free_rate
    benchmark = benchmark.reindex(returns.index).dropna() if benchmark is not None else pd.Series(dtype=float)
    beta = alpha = tracking_error = 0.0
    if not benchmark.empty:
        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        if len(aligned) > 1 and aligned.iloc[:, 1].var() != 0:
            beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
            alpha = float(aligned.iloc[:, 0].mean() * 252 - beta * aligned.iloc[:, 1].mean() * 252)
            tracking_error = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).std(ddof=0) * np.sqrt(252))
    return {
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(excess / volatility) if volatility else 0.0,
        "sortino": float(excess / downside) if downside else 0.0,
        "max_drawdown": max_drawdown(curve),
        "volatility": volatility,
        "cumulative_return": float(curve.iloc[-1] - 1),
        "alpha": alpha,
        "beta": beta,
        "tracking_error": tracking_error,
    }


def simple_equal_weight_backtest(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Create a daily equal-weight return series for selected tickers."""
    if not tickers or prices.empty:
        return pd.DataFrame(columns=["date", "portfolio_return", "equity"])
    pivot = prices[prices["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns = pivot.pct_change().mean(axis=1).fillna(0)
    return pd.DataFrame({"date": returns.index, "portfolio_return": returns.values, "equity": equity_curve(returns).values})
