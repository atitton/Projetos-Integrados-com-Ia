"""End-to-end orchestration for data update, scoring, portfolio and exports."""
from __future__ import annotations

from datetime import date
import logging

import pandas as pd

from config import Settings
from modules.backtest import performance_metrics, simple_equal_weight_backtest
from modules.data_sources import fetch_brapi_fundamentals, fetch_prices
from modules.database import SQLiteRepository
from modules.exporters import export_excel, export_pdf_report
from modules.factors import annualized_volatility, build_equal_weight_portfolio, calculate_indicators

LOGGER = logging.getLogger(__name__)
DEFAULT_TICKERS = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "ABEV3", "RENT3", "PRIO3", "SUZB3", "ELET3", "EQTL3", "GGBR4", "RADL3", "VIVT3", "B3SA3", "JBSS3", "LREN3", "CSAN3", "RAIL3", "KLBN11", "TIMS3", "CMIG4", "CPLE6"]


def run_pipeline(settings: Settings, tickers: list[str] | None = None) -> dict[str, object]:
    """Run the complete quantitative value investing workflow."""
    tickers = tickers or DEFAULT_TICKERS
    repo = SQLiteRepository(settings.database_path)
    prices = fetch_prices(tickers, settings.start_date)
    fundamentals = fetch_brapi_fundamentals(tickers, settings.brapi_token)
    if not prices.empty:
        repo.upsert_dataframe("prices", prices, ["ticker", "date"])
    if not fundamentals.empty:
        repo.upsert_dataframe("fundamentals", fundamentals, ["ticker", "reference_date"])
    prices = repo.read_table("prices")
    fundamentals = repo.read_table("fundamentals")
    risk = annualized_volatility(prices, settings.trading_days)
    indicators = calculate_indicators(fundamentals, risk, settings.min_average_volume)
    indicators["date"] = date.today().isoformat()
    portfolio = build_equal_weight_portfolio(indicators, settings.portfolio_size)
    if not indicators.empty:
        repo.upsert_dataframe("indicators", indicators[["ticker", "date", "earnings_yield", "cash_flow_yield", "book_to_market", "volatility", "z_ey", "z_cfy", "z_btm", "score", "exclusion_reason"]], ["ticker", "date"])
    if not portfolio.empty:
        portfolio_db = portfolio.copy(); portfolio_db["rebalance_date"] = date.today().isoformat()
        repo.upsert_dataframe("portfolio", portfolio_db[["rebalance_date", "ticker", "weight", "price", "score"]], ["rebalance_date", "ticker"])
    bt = simple_equal_weight_backtest(prices, portfolio["ticker"].tolist() if not portfolio.empty else [])
    returns = bt.set_index("date")["portfolio_return"] if not bt.empty else pd.Series(dtype=float)
    metrics = performance_metrics(returns, risk_free_rate=settings.risk_free_rate)
    export_excel(settings.output_dir, indicators, portfolio, indicators)
    export_pdf_report(settings.output_dir, metrics)
    return {"ranking": indicators, "portfolio": portfolio, "backtest": bt, "metrics": metrics}
