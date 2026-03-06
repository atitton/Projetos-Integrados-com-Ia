import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class MT5Config:
    timezone: str = "America/Sao_Paulo"


class MT5ConnectionError(Exception):
    """Erro ao conectar ou obter dados do MetaTrader5."""


class MT5Connector:
    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config or MT5Config()
        self._mt5 = None

    def _load_mt5(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5  # type: ignore

            self._mt5 = mt5
            return mt5
        except Exception as exc:  # pragma: no cover - depende de ambiente
            raise MT5ConnectionError("Não foi possível importar MetaTrader5.") from exc

    def connect(self):
        mt5 = self._load_mt5()
        if not mt5.initialize():  # pragma: no cover - depende de ambiente
            error = mt5.last_error()
            raise MT5ConnectionError(f"Falha ao inicializar MT5: {error}")
        LOGGER.info("Conectado ao MetaTrader5 com sucesso.")

    def shutdown(self):
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()  # pragma: no cover - depende de ambiente
            except Exception:
                LOGGER.exception("Falha ao encerrar conexão MT5.")

    def fetch_ohlcv(self, symbol: str, start: str, end: str, timeframe: Optional[int] = None) -> pd.DataFrame:
        mt5 = self._load_mt5()
        self.connect()

        tf = timeframe if timeframe is not None else mt5.TIMEFRAME_D1
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            raise MT5ConnectionError(f"Nenhum dado retornado para {symbol} entre {start} e {end}.")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(self.config.timezone)
        df = df.rename(columns={"time": "date", "tick_volume": "volume"})
        expected_cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[expected_cols].sort_values("date").reset_index(drop=True)
        LOGGER.info("Dados OHLCV extraídos do MT5: %s linhas.", len(df))
        return df
