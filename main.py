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
    """Gera uma interpretação objetiva de mercado a partir das métricas."""
    slope = np.nan
    if len(term_df) >= 2:
        slope = (term_df["atm_iv"].iloc[-1] - term_df["atm_iv"].iloc[0]) / (
            term_df["days_to_expiration"].iloc[-1] - term_df["days_to_expiration"].iloc[0]
        )

    if avg_iv > 0.45:
        vol_level = "Volatilidade aparenta estar cara em termos absolutos"
    elif avg_iv < 0.20:
        vol_level = "Volatilidade aparenta estar relativamente barata"
    else:
        vol_level = "Volatilidade em faixa neutra"

    if np.isnan(avg_skew):
        skew_text = "Skew inconclusivo por baixa cobertura de opções OTM"
    elif avg_skew > 0.03:
        skew_text = "Skew de proteção na queda está elevado (mercado pagando por puts)"
    elif avg_skew < -0.03:
        skew_text = "Skew de alta domina (calls relativamente mais caras)"
    else:
        skew_text = "Skew equilibrado"

    if np.isnan(slope):
        term_text = "Inclinação da estrutura a termo indisponível"
    elif slope > 0:
        term_text = "Estrutura a termo em contango de volatilidade"
    elif slope < 0:
        term_text = "Estrutura a termo em backwardation de volatilidade"
    else:
        term_text = "Estrutura a termo está plana"

    return f"{vol_level}. {skew_text}. {term_text}."


def run_analysis(input_csv: str, output_dir: str, risk_free_rate: float) -> dict[str, pd.DataFrame]:
    """Executa o fluxo completo de volatilidade de opções e exporta gráficos + tabelas."""
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
    print("RESUMO DO DASHBOARD DE VOLATILIDADE DE OPÇÕES")
    print("=" * 72)
    print(f"Linhas carregadas: {len(df):,}")
    print(f"IV média: {avg_iv:.4f}")
    print(f"Skew simples médio (put OTM - call OTM): {avg_skew:.4f}")
    print(f"Interpretação: {interpretation}")
    print("Arquivos de saída:")
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
    """Faz o parse dos argumentos da CLI."""
    parser = argparse.ArgumentParser(description="Análise de volatilidade implícita, skew e superfície")
    parser.add_argument("--input_csv", required=True, help="Caminho do CSV de opções de entrada")
    parser.add_argument("--output_dir", default="outputs", help="Diretório para gráficos e CSVs derivados")
    parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.10,
        help="Taxa livre de risco anual padrão se ausente no CSV (decimal, ex: 0.10)",
    )
    return parser.parse_args()


def main() -> None:
    """Executor da CLI."""
    args = parse_args()
    run_analysis(args.input_csv, args.output_dir, args.risk_free_rate)


if __name__ == "__main__":
    main()
