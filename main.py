"""Entrypoint unificado para pipeline DV (legado) e analytics de opções."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

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
from dv_calculator import DVConfig, build_dv_levels
from macro_data import MacroDataFetcher, merge_market_and_macro
from ml_model import ATRModel
from mt5_connection import MT5ConnectionError, MT5Connector
from technical_indicators import add_technical_features
from visualization import plot_term_structure, plot_vol_smile, plot_vol_surface


def setup_logging() -> None:
    """Configura logging padrão para execução via CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# =========================
# Pipeline DV (compatível)
# =========================
def prepare_dataset(symbol: str, start: str, end: str, atr_period: int) -> pd.DataFrame:
    """Monta dataset de mercado + macro + features técnicas para o pipeline DV."""
    connector = MT5Connector()
    market = connector.fetch_ohlcv(symbol=symbol, start=start, end=end)

    macro_fetcher = MacroDataFetcher()
    macro = macro_fetcher.build_macro_frame(start=start, end=end)

    merged = merge_market_and_macro(market, macro)
    features = add_technical_features(merged, atr_period=atr_period)
    features["atr_target"] = features["atr"].shift(-1)
    return features


def run_pipeline(
    symbol: str,
    start: str,
    end: str,
    use_ml: bool,
    atr_period: int,
    multipliers: List[int],
    output_path: str,
) -> pd.DataFrame:
    """Executa pipeline DV legado (mantido para compatibilidade com testes e integrações)."""
    df = prepare_dataset(symbol, start, end, atr_period)

    feature_cols = [c for c in df.columns if c not in {"date", "atr", "atr_target"} and df[c].dtype != "O"]

    if use_ml:
        model = ATRModel()
        model.train(df, feature_cols=feature_cols, target_col="atr_target")
        df["atr_pred"] = model.predict(df)
        model.save()
    else:
        df["atr_pred"] = df["atr"]

    dv = build_dv_levels(df.dropna(subset=["atr_pred", "close"]), DVConfig(multipliers=multipliers))

    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dv.to_csv(output_csv, index=False)

    output_xlsx = output_csv.with_suffix(".xlsx")
    dv.to_excel(output_xlsx, index=False)

    logging.info("Arquivos DV exportados: %s e %s", output_csv, output_xlsx)
    return dv


# =================================
# Pipeline de opções (novo workflow)
# =================================
def _market_interpretation(avg_iv: float, avg_skew: float, term_df: pd.DataFrame) -> str:
    """Gera uma interpretação objetiva de mercado a partir das métricas de opções."""
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


def run_options_analysis(input_csv: str, output_dir: str, risk_free_rate: float) -> dict[str, pd.DataFrame]:
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
    """Faz o parse dos argumentos da CLI (modo DV legado ou modo opções)."""
    parser = argparse.ArgumentParser(description="Ferramenta unificada: DV (legado) e opções")
    subparsers = parser.add_subparsers(dest="mode")

    # Modo DV legado
    dv_parser = subparsers.add_parser("dv", help="Executa pipeline DV legado (MT5 + macro + ATR)")
    dv_parser.add_argument("--symbol", required=True, help="Símbolo no MT5, ex: WIN$N")
    dv_parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD")
    dv_parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD")
    dv_parser.add_argument("--use_ml", action="store_true", help="Se informado, usa modelo ML para prever ATR+1")
    dv_parser.add_argument("--atr_period", type=int, default=14, help="Período ATR")
    dv_parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 3, 4], help="Multiplicadores DV")
    dv_parser.add_argument("--output_path", default="dv_resultados.csv", help="Caminho do CSV de saída")

    # Modo opções
    opt_parser = subparsers.add_parser("options", help="Executa analytics de volatilidade para opções")
    opt_parser.add_argument("--input_csv", required=True, help="Caminho do CSV de opções de entrada")
    opt_parser.add_argument("--output_dir", default="outputs", help="Diretório para gráficos e CSVs derivados")
    opt_parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.10,
        help="Taxa livre de risco anual padrão se ausente no CSV (decimal, ex: 0.10)",
    )

    # Compatibilidade: se nenhum subcomando for informado, assumimos pipeline DV legado.
    parser.set_defaults(mode="dv")
    return parser.parse_args()


def main() -> None:
    """Executor principal da CLI."""
    setup_logging()
    args = parse_args()

    try:
        if args.mode == "options":
            run_options_analysis(args.input_csv, args.output_dir, args.risk_free_rate)
        else:
            run_pipeline(
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                use_ml=args.use_ml,
                atr_period=args.atr_period,
                multipliers=args.multipliers,
                output_path=args.output_path,
            )
    except MT5ConnectionError as exc:
        logging.exception("Falha na conexão com MT5: %s", exc)
        raise SystemExit(2) from exc
    except Exception as exc:
        logging.exception("Erro durante execução do pipeline: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
