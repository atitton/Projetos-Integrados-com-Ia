# Projeto DV com ATR (MT5 + Macro + ML)

Pipeline completo para gerar níveis DV diários a partir de dados reais de mercado via MetaTrader5, enriquecidos com dados macroeconômicos, indicadores técnicos e previsão de ATR.

## Estrutura

- `mt5_connection.py`: conexão MT5 e extração OHLCV.
- `macro_data.py`: ingestão de macro dados (BCB, Yahoo, FRED) e merge com mercado.
- `technical_indicators.py`: cálculo de ATR, RSI, MACD e features técnicas.
- `ml_model.py`: treino, persistência e inferência do modelo de previsão de ATR.
- `dv_calculator.py`: cálculo de `target_date`, `ref_price` e níveis DV.
- `main.py`: CLI principal.
- `tests/`: testes com Pytest.
- `requirements.txt`: dependências.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Se usar FRED, configure `FRED_API_KEY` no ambiente (ou `.env`).

## Execução CLI

```bash
python main.py --symbol "WIN$N" --start "2023-01-01" --end "2024-03-06" --use_ml --atr_period 14 --multipliers 1 2 3 4 --output_path "dv_resultados.csv"
```

Saídas geradas:

- `dv_resultados.csv`
- `dv_resultados.xlsx`
- `artifacts/atr_model.pkl` (quando `--use_ml`)

## Colunas de saída

- `calculation_date`
- `target_date` (próximo dia útil considerando finais de semana + feriados fixos)
- `ref_price` (`close_D`)
- `atr_pred`
- `dv_plus_1` a `dv_plus_4`
- `dv_minus_1` a `dv_minus_4`

## Tratamento de erros e logging

- Erros de MT5 são capturados com exceção dedicada (`MT5ConnectionError`).
- Fontes macro são tentadas de forma resiliente e consolidadas com preenchimento (`ffill/bfill`).
- Logging informativo habilitado em `main.py`.

## Testes

```bash
pytest -q
```

Coberturas principais:

- Cálculo de ATR com série conhecida.
- Previsão de ATR com modelo fake.
- Integração ponta-a-ponta (MT5 + macro + previsão + exportação), usando mocks.
