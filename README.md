# Sistema de Análise de Volatilidade de Opções

Toolkit Python orientado a produção para analisar volatilidade implícita, skew, estrutura a termo e superfície de volatilidade a partir de CSV de opções.

## Módulos

- `data_loader.py`: ingestão do CSV, validação de esquema, limpeza e criação de features (`time_to_expiration`, `log_moneyness`).
- `black_scholes.py`: precificação Black-Scholes e vega para calls e puts.
- `implied_vol.py`: solver robusto de volatilidade implícita (Newton-Raphson com fallback de Brent).
- `analytics.py`: pipeline de IV, métricas de skew, estrutura a termo ATM, detecção de regime e anomalias.
- `visualization.py`: gráficos de smile, estrutura a termo ATM e superfície 3D (saída HTML com Plotly).
- `main.py`: ponto de entrada CLI e resumo executivo no console.

## Formato esperado do CSV de entrada

- `underlying_price`
- `option_type` (`call`/`put`)
- `strike`
- `expiration_date`
- `days_to_expiration`
- `option_price`
- `risk_free_rate` (opcional; usa default da CLI quando ausente)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py \
  --input_csv data/options_quotes.csv \
  --output_dir outputs \
  --risk_free_rate 0.105
```

## Saídas

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

## Como interpretar (visão de mesa)

O dashboard ajuda a responder:

- Onde a volatilidade está cara/barata (nível absoluto de IV + perfil temporal)
- Se o mercado teme mais queda ou alta (skew)
- Se há estresse de curto prazo (spikes na ponta curta)
