from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from dv_calculator import DVConfig, build_dv_levels
from macro_data import MacroDataFetcher, merge_market_and_macro
from ml_model import ATRModel
from mt5_connection import MT5ConnectionError, MT5Connector
from technical_indicators import add_technical_features


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def prepare_dataset(symbol: str, start: str, end: str, atr_period: int) -> pd.DataFrame:
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
):
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

    logging.info("Arquivos exportados: %s e %s", output_csv, output_xlsx)
    return dv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geração de níveis DV com previsão de ATR.")
    parser.add_argument("--symbol", required=True, help="Símbolo no MT5, ex: WIN$N")
    parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD")
    parser.add_argument("--use_ml", action="store_true", help="Se informado, usa modelo ML para prever ATR+1")
    parser.add_argument("--atr_period", type=int, default=14, help="Período ATR")
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 3, 4], help="Multiplicadores DV")
    parser.add_argument("--output_path", default="dv_resultados.csv", help="Caminho do CSV de saída")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    try:
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
