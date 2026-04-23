"""CLI entrypoint for options implied volatility and skew analytics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import (
    build_term_structure,
    compute_implied_vols,
    compute_skew_metrics,
    detect_vol_regime,
    flag_anomalies,
)
from data_loader import LoaderConfig, load_options_csv
from visualization import plot_term_structure, plot_vol_smile, plot_vol_surface


def _market_interpretation(avg_iv: float, avg_skew: float, term_df: pd.DataFrame) -> str:
    """Generate a concise trading interpretation from computed metrics."""
    slope = np.nan
    if len(term_df) >= 2:
        slope = (term_df["atm_iv"].iloc[-1] - term_df["atm_iv"].iloc[0]) / (
            term_df["days_to_expiration"].iloc[-1] - term_df["days_to_expiration"].iloc[0]
        )

    if avg_iv > 0.45:
        vol_level = "Volatility looks expensive in absolute terms"
    elif avg_iv < 0.20:
        vol_level = "Volatility looks relatively cheap"
    else:
        vol_level = "Volatility is around a neutral range"

    if np.isnan(avg_skew):
        skew_text = "Skew is inconclusive due to limited OTM coverage"
    elif avg_skew > 0.03:
        skew_text = "Downside skew is elevated (market paying for puts)"
    elif avg_skew < -0.03:
        skew_text = "Upside skew dominates (calls relatively rich)"
    else:
        skew_text = "Skew is balanced"

    if np.isnan(slope):
        term_text = "Term-structure slope unavailable"
    elif slope > 0:
        term_text = "Term structure is in vol contango"
    elif slope < 0:
        term_text = "Term structure is in vol backwardation"
    else:
        term_text = "Term structure is flat"

    return f"{vol_level}. {skew_text}. {term_text}."


def run_analysis(input_csv: str, output_dir: str, risk_free_rate: float) -> dict[str, pd.DataFrame]:
    """Run full options volatility workflow and export plots + tables."""
    df = load_options_csv(input_csv, LoaderConfig(risk_free_rate_default=risk_free_rate))
    df_iv = compute_implied_vols(df)
    skew_df = compute_skew_metrics(df_iv)
    term_df = build_term_structure(df_iv)
    regime_df = detect_vol_regime(df_iv)
    anomalies = flag_anomalies(skew_df, term_df)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    smile_path = plot_vol_smile(df_iv, output_dir=output_dir)
    term_path = plot_term_structure(term_df, output_dir=output_dir)
    surface_path = plot_vol_surface(df_iv, output_dir=output_dir)

    df_iv.to_csv(outdir / "options_with_iv.csv", index=False)
    skew_df.to_csv(outdir / "skew_metrics.csv", index=False)
    term_df.to_csv(outdir / "term_structure.csv", index=False)
    regime_df.to_csv(outdir / "vol_regime.csv", index=False)
    anomalies["skew_anomalies"].to_csv(outdir / "skew_anomalies.csv", index=False)
    anomalies["short_term_iv_spikes"].to_csv(outdir / "short_term_iv_spikes.csv", index=False)

    avg_iv = float(df_iv["implied_vol"].mean(skipna=True))
    avg_skew = float(skew_df["simple_skew"].mean(skipna=True))
    interpretation = _market_interpretation(avg_iv, avg_skew, term_df)

    print("=" * 72)
    print("OPTIONS VOLATILITY DASHBOARD SUMMARY")
    print("=" * 72)
    print(f"Rows loaded: {len(df):,}")
    print(f"Average IV: {avg_iv:.4f}")
    print(f"Average simple skew (put OTM - call OTM): {avg_skew:.4f}")
    print(f"Interpretation: {interpretation}")
    print("Output files:")
    print(f"  - Smile: {smile_path}")
    print(f"  - Term Structure: {term_path}")
    print(f"  - Surface: {surface_path}")

    return {
        "options_with_iv": df_iv,
        "skew_metrics": skew_df,
        "term_structure": term_df,
        "vol_regime": regime_df,
        "skew_anomalies": anomalies["skew_anomalies"],
        "short_term_iv_spikes": anomalies["short_term_iv_spikes"],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Implied volatility, skew, and surface analytics")
    parser.add_argument("--input_csv", required=True, help="Input options CSV path")
    parser.add_argument("--output_dir", default="outputs", help="Directory for plots and derived CSV files")
    parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.10,
        help="Default annualized risk-free rate if missing in input (decimal, e.g. 0.10)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI runner."""
    args = parse_args()
    run_analysis(args.input_csv, args.output_dir, args.risk_free_rate)


if __name__ == "__main__":
    main()
