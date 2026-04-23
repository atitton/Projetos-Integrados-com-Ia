"""Data loading and preprocessing utilities for options volatility analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "underlying_price",
    "option_type",
    "strike",
    "expiration_date",
    "days_to_expiration",
    "option_price",
}


@dataclass
class LoaderConfig:
    """Configuration for loading and validating options CSV data."""

    risk_free_rate_default: float = 0.10
    min_option_price: float = 1e-6


def _normalize_option_type(series: pd.Series) -> pd.Series:
    """Normalize option type labels to {'call', 'put'} when possible."""
    mapping = {
        "c": "call",
        "call": "call",
        "calls": "call",
        "p": "put",
        "put": "put",
        "puts": "put",
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    return normalized


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_options_csv(path: str | Path, config: LoaderConfig | None = None) -> pd.DataFrame:
    """Load options data from CSV, clean columns, and enforce numeric consistency.

    Parameters
    ----------
    path
        CSV file path.
    config
        Optional loader configuration.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for implied-volatility calculations.
    """
    cfg = config or LoaderConfig()
    df = pd.read_csv(path)
    _ensure_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["option_type"] = _normalize_option_type(df["option_type"])
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")

    if "risk_free_rate" not in df.columns:
        df["risk_free_rate"] = cfg.risk_free_rate_default
    df["risk_free_rate"] = pd.to_numeric(df["risk_free_rate"], errors="coerce").fillna(cfg.risk_free_rate_default)

    numeric_cols = [
        "underlying_price",
        "strike",
        "days_to_expiration",
        "option_price",
        "risk_free_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "underlying_price",
            "option_type",
            "strike",
            "expiration_date",
            "days_to_expiration",
            "option_price",
            "risk_free_rate",
        ]
    )

    df = df[df["option_type"].isin(["call", "put"])]
    df = df[(df["underlying_price"] > 0) & (df["strike"] > 0)]
    df = df[df["days_to_expiration"] >= 0]
    df = df[df["option_price"] >= cfg.min_option_price]

    df["time_to_expiration"] = np.maximum(df["days_to_expiration"] / 365.0, 1e-8)
    df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
    return df.reset_index(drop=True)
