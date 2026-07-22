from __future__ import annotations

import pandas as pd

from modules.backtest import performance_metrics
from modules.factors import annualized_volatility, build_equal_weight_portfolio, calculate_indicators


def test_factor_scoring_and_exclusion_reasons() -> None:
    fundamentals = pd.DataFrame([
        {"ticker": "AAA3", "reference_date": "2026-01-01", "ebit": 100.0, "enterprise_value": 1000.0, "operating_cash_flow": 120.0, "equity": 800.0, "market_cap": 1200.0, "net_income": 90.0, "judicial_recovery": 0, "opa": 0},
        {"ticker": "BBB3", "reference_date": "2026-01-01", "ebit": 50.0, "enterprise_value": 1000.0, "operating_cash_flow": 40.0, "equity": -10.0, "market_cap": 500.0, "net_income": -1.0, "judicial_recovery": 0, "opa": 0},
    ])
    risk = pd.DataFrame([
        {"ticker": "AAA3", "volatility": 0.2, "avg_financial_volume_3m": 10_000_000.0, "price": 10.0},
        {"ticker": "BBB3", "volatility": 0.3, "avg_financial_volume_3m": 1_000_000.0, "price": 5.0},
    ])
    indicators = calculate_indicators(fundamentals, risk, 6_000_000.0)
    bbb = indicators[indicators["ticker"].eq("BBB3")].iloc[0]
    assert "lucro líquido negativo" in bbb["exclusion_reason"]
    assert "patrimônio líquido negativo" in bbb["exclusion_reason"]
    portfolio = build_equal_weight_portfolio(indicators, 20)
    assert portfolio["ticker"].tolist() == ["AAA3"]
    assert portfolio["weight"].iloc[0] == 1.0


def test_annualized_volatility_and_metrics() -> None:
    prices = pd.DataFrame({
        "ticker": ["AAA3"] * 4,
        "date": pd.date_range("2026-01-01", periods=4).strftime("%Y-%m-%d"),
        "adj_close": [10.0, 11.0, 10.5, 12.0],
        "close": [10.0, 11.0, 10.5, 12.0],
        "financial_volume": [7_000_000.0] * 4,
    })
    risk = annualized_volatility(prices)
    assert risk.loc[0, "volatility"] > 0
    metrics = performance_metrics(pd.Series([0.01, -0.005, 0.02]))
    assert set(metrics) == {"cagr", "sharpe", "sortino", "max_drawdown", "volatility", "cumulative_return", "alpha", "beta", "tracking_error"}
