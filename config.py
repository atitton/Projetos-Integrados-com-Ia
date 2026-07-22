"""Central configuration for the Brazilian value investing system."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    database_path: Path = Path(os.getenv("DATABASE_PATH", "database/value_investing.db"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    brapi_token: str | None = os.getenv("BRAPI_TOKEN")
    start_date: str = os.getenv("START_DATE", "2004-01-01")
    min_average_volume: float = float(os.getenv("MIN_AVERAGE_VOLUME", "6000000"))
    portfolio_size: int = int(os.getenv("PORTFOLIO_SIZE", "20"))
    trading_days: int = int(os.getenv("TRADING_DAYS", "252"))
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.105"))
    benchmark_ibov: str = os.getenv("BENCHMARK_IBOV", "^BVSP")
    benchmark_idiv: str = os.getenv("BENCHMARK_IDIV", "IDIV.SA")


settings = Settings()
