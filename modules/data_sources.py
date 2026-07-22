"""Data acquisition from yfinance, BRAPI, CVM and Fundamentus fallback."""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)
BRAPI_BASE = "https://brapi.dev/api"


def normalize_ticker(ticker: str) -> str:
    """Return a B3 ticker in yfinance format."""
    return ticker if ticker.endswith(".SA") or ticker.startswith("^") else f"{ticker}.SA"


def fetch_prices(tickers: Iterable[str], start: str) -> pd.DataFrame:
    """Download daily OHLCV data and financial volume from yfinance."""
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        yf_ticker = normalize_ticker(ticker)
        try:
            raw = yf.download(yf_ticker, start=start, progress=False, auto_adjust=False, threads=False)
        except Exception as exc:  # noqa: BLE001 - log source failures without stopping the batch
            LOGGER.warning("Could not download prices for %s: %s", ticker, exc)
            continue
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
        frame["ticker"] = ticker.replace(".SA", "")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["financial_volume"] = frame["close"] * frame["volume"]
        rows.append(frame[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "financial_volume"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_brapi_fundamentals(tickers: Iterable[str], token: str | None = None) -> pd.DataFrame:
    """Fetch available fundamental fields from BRAPI quote endpoint."""
    records: list[dict[str, object]] = []
    for ticker in tickers:
        params = {"fundamental": "true"}
        if token:
            params["token"] = token
        try:
            response = requests.get(f"{BRAPI_BASE}/quote/{ticker.replace('.SA', '')}", params=params, timeout=20)
            response.raise_for_status()
            result = response.json().get("results", [{}])[0]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("BRAPI failed for %s: %s", ticker, exc)
            continue
        records.append({
            "ticker": ticker.replace(".SA", ""),
            "reference_date": date.today().isoformat(),
            "ebit": result.get("ebit"),
            "ebitda": result.get("ebitda"),
            "enterprise_value": result.get("enterpriseValue"),
            "operating_cash_flow": result.get("operatingCashflow"),
            "equity": result.get("bookValue") and result.get("bookValue") * result.get("numberOfShares", np.nan),
            "market_cap": result.get("marketCap"),
            "revenue": result.get("totalRevenue"),
            "net_income": result.get("netIncomeToCommon"),
            "dividend_yield": result.get("dividendYield"),
            "roe": result.get("returnOnEquity"),
            "roic": result.get("returnOnCapital"),
            "net_debt": result.get("netDebt"),
            "cash": result.get("totalCash"),
            "disclosure_date": result.get("earningsTimestamp") or date.today().isoformat(),
            "judicial_recovery": 0,
            "opa": 0,
        })
    return pd.DataFrame.from_records(records)


def fetch_fundamentus_status(ticker: str) -> dict[str, bool]:
    """Scrape public status flags when no API field is available."""
    url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker.replace('.SA', '')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers, timeout=20).text
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Fundamentus status failed for %s: %s", ticker, exc)
        return {"judicial_recovery": False, "opa": False}
    text = BeautifulSoup(html, "html.parser").get_text(" ").lower()
    return {"judicial_recovery": "recuperação judicial" in text, "opa": "opa" in text}


def fetch_cvm_disclosures() -> pd.DataFrame:
    """Placeholder for CVM disclosure metadata integration.

    CVM bulk datasets change by year and filing type; this function provides the
    integration point used by the pipeline while keeping the system runnable when
    network files are unavailable.
    """
    return pd.DataFrame(columns=["ticker", "disclosure_date", "reference_date"])
