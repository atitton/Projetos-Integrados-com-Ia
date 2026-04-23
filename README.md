# Options Volatility Analytics System

Production-oriented Python toolkit to analyze implied volatility, skew, term structure, and volatility surface from options CSV data.

## Modules

- `data_loader.py`: CSV ingestion, schema validation, cleaning, and feature prep (`time_to_expiration`, `log_moneyness`).
- `black_scholes.py`: Black-Scholes pricing + vega for calls and puts.
- `implied_vol.py`: Robust IV solver (Newton-Raphson with Brent fallback).
- `analytics.py`: IV computation pipeline, skew metrics, term structure, regime/anomaly flags.
- `visualization.py`: Smile, ATM term structure, and 3D volatility surface (Plotly HTML outputs).
- `main.py`: CLI entrypoint and console summary.

## Expected Input CSV Columns

- `underlying_price`
- `option_type` (`call`/`put`)
- `strike`
- `expiration_date`
- `days_to_expiration`
- `option_price`
- `risk_free_rate` (optional; default from CLI)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
  --input_csv data/options_quotes.csv \
  --output_dir outputs \
  --risk_free_rate 0.105
```

## Outputs

Inside `output_dir`:

- `options_with_iv.csv`
- `skew_metrics.csv`
- `term_structure.csv`
- `vol_regime.csv`
- `skew_anomalies.csv`
- `short_term_iv_spikes.csv`
- `vol_smile.html`
- `term_structure.html`
- `vol_surface.html`

## Notes for Traders

The dashboard helps answer:

- Where volatility is expensive/cheap (absolute IV + term profile)
- Whether downside or upside fear dominates (skew)
- If short-end volatility stress is present (spikes in short maturities)
