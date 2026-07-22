"""Factor engineering, filters, z-scores and portfolio construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(prices: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Calculate trailing annualized volatility per ticker."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["return"] = df.groupby("ticker")["adj_close"].pct_change()
    vol = df.groupby("ticker")["return"].std(ddof=0).mul(np.sqrt(trading_days)).reset_index(name="volatility")
    avg_volume = df.groupby("ticker").tail(63).groupby("ticker")["financial_volume"].mean().reset_index(name="avg_financial_volume_3m")
    last_price = df.groupby("ticker").tail(1)[["ticker", "close"]].rename(columns={"close": "price"})
    return vol.merge(avg_volume, on="ticker", how="outer").merge(last_price, on="ticker", how="outer")


def calculate_indicators(fundamentals: pd.DataFrame, risk: pd.DataFrame, min_average_volume: float) -> pd.DataFrame:
    """Create valuation factors, exclusion reasons and composite score."""
    df = fundamentals.copy().merge(risk, on="ticker", how="left")
    df["earnings_yield"] = df["ebit"] / df["enterprise_value"]
    df["cash_flow_yield"] = df["operating_cash_flow"] / df["enterprise_value"]
    df["book_to_market"] = df["equity"] / df["market_cap"]
    reasons: list[str] = []
    vol_cut = df["volatility"].quantile(0.9)
    for row in df.itertuples(index=False):
        row_reasons: list[str] = []
        if pd.isna(row.avg_financial_volume_3m) or row.avg_financial_volume_3m < min_average_volume:
            row_reasons.append("volume médio diário abaixo de R$ 6 milhões")
        if pd.notna(row.net_income) and row.net_income < 0:
            row_reasons.append("lucro líquido negativo")
        if bool(row.judicial_recovery):
            row_reasons.append("recuperação judicial")
        if bool(row.opa):
            row_reasons.append("OPA")
        if pd.notna(row.equity) and row.equity < 0:
            row_reasons.append("patrimônio líquido negativo")
        if pd.notna(row.volatility) and pd.notna(vol_cut) and row.volatility >= vol_cut:
            row_reasons.append("decil mais volátil")
        reasons.append("; ".join(row_reasons))
    df["exclusion_reason"] = reasons
    eligible = df["exclusion_reason"].eq("")
    for factor, z_col in [("earnings_yield", "z_ey"), ("cash_flow_yield", "z_cfy"), ("book_to_market", "z_btm")]:
        mean = df.loc[eligible, factor].mean()
        std = df.loc[eligible, factor].std(ddof=0)
        df[z_col] = np.where(eligible & pd.notna(std) & (std != 0), (df[factor] - mean) / std if pd.notna(std) and std != 0 else np.nan, np.nan)
    df["score"] = df[["z_ey", "z_cfy", "z_btm"]].sum(axis=1, min_count=3)
    return df.sort_values("score", ascending=False, na_position="last")


def build_equal_weight_portfolio(indicators: pd.DataFrame, size: int = 20) -> pd.DataFrame:
    """Select the top-scoring eligible stocks with equal weights."""
    selected = indicators[indicators["exclusion_reason"].eq("")].nlargest(size, "score").copy()
    if selected.empty:
        return selected
    selected["weight"] = 1 / len(selected)
    return selected[["ticker", "weight", "price", "score"]]
