import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass
class MacroConfig:
    timezone: str = "America/Sao_Paulo"


class MacroDataError(Exception):
    """Erro ao obter ou processar dados macroeconômicos."""


class MacroDataFetcher:
    def __init__(self, config: Optional[MacroConfig] = None):
        load_dotenv()
        self.config = config or MacroConfig()

    def fetch_bcb_selic(self, start: str, end: str) -> pd.DataFrame:
        url = (
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
            f"?formato=json&dataInicial={pd.to_datetime(start).strftime('%d/%m/%Y')}"
            f"&dataFinal={pd.to_datetime(end).strftime('%d/%m/%Y')}"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            if df.empty:
                raise MacroDataError("BCB retornou dados vazios para Selic.")
            df["date"] = pd.to_datetime(df["data"], dayfirst=True).dt.tz_localize(self.config.timezone)
            df["selic"] = pd.to_numeric(df["valor"], errors="coerce")
            return df[["date", "selic"]]
        except Exception as exc:
            raise MacroDataError("Falha ao baixar Selic do BCB.") from exc

    def fetch_yahoo_series(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                raise MacroDataError(f"Yahoo retornou série vazia para {ticker}.")
            df = raw[["Close"]].rename(columns={"Close": f"{ticker}_close"}).reset_index()
            df["date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(self.config.timezone)
            return df[["date", f"{ticker}_close"]]
        except Exception as exc:
            raise MacroDataError(f"Falha ao baixar dados do Yahoo para {ticker}.") from exc

    def fetch_fred_series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        try:
            from fredapi import Fred

            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            if series.empty:
                raise MacroDataError(f"FRED retornou série vazia para {series_id}.")
            df = series.rename(series_id).reset_index().rename(columns={"index": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(self.config.timezone)
            return df
        except Exception as exc:
            raise MacroDataError(f"Falha ao baixar dados FRED ({series_id}).") from exc

    def build_macro_frame(self, start: str, end: str) -> pd.DataFrame:
        frames = []
        errors = []

        for getter in [
            lambda: self.fetch_bcb_selic(start, end),
            lambda: self.fetch_yahoo_series("^GSPC", start, end),
            lambda: self.fetch_fred_series("DGS10", start, end),
        ]:
            try:
                frames.append(getter())
            except Exception as exc:
                errors.append(str(exc))

        if not frames:
            raise MacroDataError("Nenhuma fonte macroeconômica disponível. " + " | ".join(errors))

        base = frames[0]
        for frame in frames[1:]:
            base = base.merge(frame, on="date", how="outer")

        base = base.sort_values("date").ffill().bfill()
        LOGGER.info("Dados macro consolidados: %s linhas, %s colunas.", base.shape[0], base.shape[1])
        return base


def merge_market_and_macro(market_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        market_df.sort_values("date"),
        macro_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged.ffill().bfill()
