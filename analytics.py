"""Analytics layer: IV computation, skew metrics, term structure, and regime flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

from implied_vol import IVSolverConfig, implied_volatility


@dataclass
class SkewSummary:
    """Container for skew metrics by expiration."""

    expiration_date: pd.Timestamp
    simple_skew: float
    slope_skew: float
    n_obs: int


def compute_implied_vols(df: pd.DataFrame, iv_config: IVSolverConfig | None = None) -> pd.DataFrame:
    """Compute implied vol for each option row and append as `implied_vol`."""
    out = df.copy()
    cfg = iv_config or IVSolverConfig()
    out["implied_vol"] = out.apply(
        lambda row: implied_volatility(
            option_type=row["option_type"],
            market_price=row["option_price"],
            S=row["underlying_price"],
            K=row["strike"],
            r=row["risk_free_rate"],
            T=row["time_to_expiration"],
            config=cfg,
        ),
        axis=1,
    )
    out["moneyness"] = out["underlying_price"] / out["strike"]
    out["abs_moneyness_distance"] = np.abs(1 - out["moneyness"])
    return out


def compute_skew_metrics(df: pd.DataFrame, otm_threshold: float = 0.02) -> pd.DataFrame:
    """Compute simple and slope-based skew for each expiration.

    Simple skew definition:
        skew = IV_put_OTM - IV_call_OTM
    Positive values imply downside protection is relatively expensive.

    Slope skew:
        Linear regression slope of IV against log-moneyness.
    More negative slope generally indicates stronger downside skew.
    """
    rows: list[dict[str, Any]] = []

    for exp, group in df.dropna(subset=["implied_vol"]).groupby("expiration_date"):
        put_otm = group[(group["option_type"] == "put") & (group["strike"] < group["underlying_price"] * (1 - otm_threshold))]
        call_otm = group[(group["option_type"] == "call") & (group["strike"] > group["underlying_price"] * (1 + otm_threshold))]

        simple_skew = np.nan
        if not put_otm.empty and not call_otm.empty:
            simple_skew = put_otm["implied_vol"].mean() - call_otm["implied_vol"].mean()

        slope_skew = np.nan
        if group["log_moneyness"].nunique() >= 2 and group["implied_vol"].notna().sum() >= 3:
            reg = linregress(group["log_moneyness"], group["implied_vol"])
            slope_skew = reg.slope

        rows.append(
            {
                "expiration_date": exp,
                "simple_skew": simple_skew,
                "slope_skew": slope_skew,
                "n_obs": len(group),
            }
        )

    return pd.DataFrame(rows).sort_values("expiration_date").reset_index(drop=True)


def build_term_structure(df: pd.DataFrame, atm_band: float = 0.02) -> pd.DataFrame:
    """Build ATM term structure by averaging IV near moneyness=1 for each maturity."""
    atm = df[np.abs(df["moneyness"] - 1.0) <= atm_band].dropna(subset=["implied_vol"])

    term = (
        atm.groupby(["expiration_date", "days_to_expiration"], as_index=False)
        .agg(atm_iv=("implied_vol", "mean"), atm_count=("implied_vol", "size"))
        .sort_values("days_to_expiration")
        .reset_index(drop=True)
    )
    term["term_slope"] = term["atm_iv"].diff() / term["days_to_expiration"].diff()
    return term


def detect_vol_regime(df: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    """Flag maturities where average IV is materially above/below cross-sectional mean."""
    avg = (
        df.dropna(subset=["implied_vol"]).groupby("expiration_date", as_index=False).agg(mean_iv=("implied_vol", "mean"))
    )
    mu = avg["mean_iv"].mean()
    sigma = avg["mean_iv"].std(ddof=0)
    if sigma <= 1e-12:
        avg["iv_zscore"] = 0.0
    else:
        avg["iv_zscore"] = (avg["mean_iv"] - mu) / sigma
    avg["regime_flag"] = np.where(
        avg["iv_zscore"] >= z_threshold,
        "elevated_iv",
        np.where(avg["iv_zscore"] <= -z_threshold, "suppressed_iv", "normal_iv"),
    )
    return avg


def flag_anomalies(skew_df: pd.DataFrame, term_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Flag unusually high skew and short-term IV spikes."""
    skew = skew_df.copy()
    if skew["simple_skew"].notna().sum() > 2:
        q = skew["simple_skew"].quantile(0.9)
        skew_anom = skew[skew["simple_skew"] >= q]
    else:
        skew_anom = skew.iloc[0:0]

    term = term_df.copy()
    short_term = term[term["days_to_expiration"] <= 30]
    if not short_term.empty:
        cutoff = short_term["atm_iv"].quantile(0.9)
        term_anom = short_term[short_term["atm_iv"] >= cutoff]
    else:
        term_anom = short_term

    return {"skew_anomalies": skew_anom, "short_term_iv_spikes": term_anom}
