"""Command-line entry point for the value investing pipeline."""
from __future__ import annotations

import argparse

from config import settings
from modules.logging_config import configure_logging
from modules.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Brazilian quantitative value investing pipeline")
    parser.add_argument("--tickers", nargs="*", help="Optional B3 tickers, e.g. PETR4 VALE3")
    args = parser.parse_args()
    configure_logging(settings.log_dir)
    result = run_pipeline(settings, args.tickers)
    print(result["portfolio"])
    print(result["metrics"])


if __name__ == "__main__":
    main()
