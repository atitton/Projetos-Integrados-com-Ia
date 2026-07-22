"""SQLite persistence layer for market data, factors and portfolios."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
        financial_volume REAL,
        PRIMARY KEY (ticker, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        ticker TEXT NOT NULL,
        reference_date TEXT NOT NULL,
        ebit REAL, ebitda REAL, enterprise_value REAL, operating_cash_flow REAL,
        equity REAL, market_cap REAL, revenue REAL, net_income REAL, dividend_yield REAL,
        roe REAL, roic REAL, net_debt REAL, cash REAL, disclosure_date TEXT,
        judicial_recovery INTEGER DEFAULT 0, opa INTEGER DEFAULT 0,
        PRIMARY KEY (ticker, reference_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS indicators (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        earnings_yield REAL, cash_flow_yield REAL, book_to_market REAL,
        volatility REAL, z_ey REAL, z_cfy REAL, z_btm REAL, score REAL,
        exclusion_reason TEXT,
        PRIMARY KEY (ticker, date)
    )
    """,
    """,
    CREATE TABLE IF NOT EXISTS portfolio (
        rebalance_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        weight REAL NOT NULL,
        price REAL,
        score REAL,
        PRIMARY KEY (rebalance_date, ticker)
    )
    """,
    """,
    CREATE TABLE IF NOT EXISTS rebalance_history (
        rebalance_date TEXT PRIMARY KEY,
        selected_tickers TEXT NOT NULL,
        notes TEXT
    )
    """,
)


class SQLiteRepository:
    """Small repository wrapper around SQLite and pandas."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def initialize(self) -> None:
        with self.connect() as conn:
            for statement in SCHEMA:
                conn.execute(statement)
        LOGGER.info("SQLite schema ready at %s", self.path)

    def upsert_dataframe(self, table: str, df: pd.DataFrame, key_columns: Iterable[str]) -> None:
        """Insert or replace a dataframe into a table using SQLite primary keys."""
        if df.empty:
            return
        columns = list(df.columns)
        placeholders = ",".join("?" for _ in columns)
        assignments = ",".join(f"{col}=excluded.{col}" for col in columns if col not in set(key_columns))
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        if assignments:
            sql += f" ON CONFLICT({','.join(key_columns)}) DO UPDATE SET {assignments}"
        with self.connect() as conn:
            conn.executemany(sql, df[columns].itertuples(index=False, name=None))

    def read_table(self, table: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
