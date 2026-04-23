# Sistema de Análise Quantitativa (DV + Opções)

Projeto Python com **dois fluxos compatíveis**:

1. **Pipeline DV legado** (MT5 + macro + ATR/ML), preservado para manter compatibilidade com integrações anteriores.
2. **Analytics de Opções** (volatilidade implícita, skew, estrutura a termo e superfície).

## Módulos principais

### DV (legado)
- `dv_calculator.py`
- `macro_data.py`
- `ml_model.py`
- `mt5_connection.py`
- `technical_indicators.py`

### Opções (novo)
- `data_loader.py`: ingestão do CSV, validação de esquema, limpeza e criação de features (`time_to_expiration`, `log_moneyness`).
- `black_scholes.py`: precificação Black-Scholes e vega para calls e puts.
- `implied_vol.py`: solver robusto de volatilidade implícita (Newton-Raphson com fallback de Brent).
- `analytics.py`: pipeline de IV, métricas de skew, estrutura a termo ATM, detecção de regime e anomalias.
- `visualization.py`: gráficos de smile, estrutura a termo ATM e superfície 3D (saída HTML com Plotly).

## CLI unificada (`main.py`)

### 1) Modo DV legado
```bash
python main.py dv \
  --symbol "WIN$N" \
  --start "2024-01-01" \
  --end "2024-03-01" \
  --use_ml \
  --atr_period 14 \
  --multipliers 1 2 3 4 \
  --output_path dv_resultados.csv
```

### 2) Modo Opções
```bash
python main.py options \
  --input_csv data/options_quotes.csv \
  --output_dir outputs \
  --risk_free_rate 0.105
```

## Formato esperado do CSV de opções
- `underlying_price`
- `option_type` (`call`/`put`)
- `strike`
- `expiration_date`
- `days_to_expiration`
- `option_price`
- `risk_free_rate` (opcional; usa default da CLI quando ausente)

## Saídas do modo Opções
Dentro de `output_dir`:
- `options_with_iv.csv`
- `skew_metrics.csv`
- `term_structure.csv`
- `vol_regime.csv`
- `skew_anomalies.csv`
- `short_term_iv_spikes.csv`
- `vol_smile.html`
- `term_structure.html`
- `vol_surface.html`

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
