# Sistema Quantitativo de Value Investing para Ações Brasileiras

Projeto Python modular para coletar dados de ações brasileiras, persistir histórico em SQLite, calcular fatores quantitativos de valor, aplicar filtros de risco/liquidez, montar uma carteira equal weight e apresentar resultados em Streamlit.

> O repositório também preserva os módulos legados de análise DV/opções existentes.

## Funcionalidades

- Coleta de preços diários e volume financeiro via `yfinance`.
- Coleta de fundamentos via BRAPI, com pontos de integração para Fundamentus e CVM.
- Banco SQLite com tabelas de preços, fundamentos, indicadores, carteira e histórico de rebalanceamentos.
- Indicadores: Earnings Yield, Cash Flow Yield e Book to Market.
- Filtros: liquidez mínima, lucro líquido negativo, recuperação judicial, OPA, patrimônio líquido negativo e decil mais volátil.
- Z-score por fator e score composto `Z(EY) + Z(CFY) + Z(BTM)`.
- Carteira com as 20 melhores ações e pesos iguais.
- Backtest equal weight, métricas de CAGR, Sharpe, Sortino, drawdown máximo, volatilidade, retorno acumulado, alpha, beta e tracking error.
- Exportação de ranking, carteira e indicadores para Excel, além de relatório PDF.
- Dashboard Streamlit com ranking, carteira, backtest, indicadores e empresas excluídas.

## Estrutura

```text
app.py                  # Dashboard Streamlit
cli.py                  # Execução via terminal
config.py               # Configuração centralizada por .env
modules/                # Coleta, banco, fatores, backtest, exportação e pipeline
database/               # SQLite local
data/                   # Dados auxiliares
output/                 # Excel/PDF gerados
logs/                   # Logs da aplicação
tests/                  # Testes automatizados
```

## Instalação

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Execução

Pipeline via CLI:

```bash
python cli.py --tickers PETR4 VALE3 ITUB4 BBAS3 WEGE3
```

Dashboard:

```bash
streamlit run app.py
```

## Configuração

Variáveis disponíveis em `.env`:

- `DATABASE_PATH`
- `OUTPUT_DIR`
- `LOG_DIR`
- `BRAPI_TOKEN`
- `START_DATE`
- `MIN_AVERAGE_VOLUME`
- `PORTFOLIO_SIZE`
- `TRADING_DAYS`
- `RISK_FREE_RATE`

## Observações de dados

APIs são priorizadas. Scraping do Fundamentus fica isolado em `modules/data_sources.py` e deve ser usado apenas para campos indisponíveis por API. A integração CVM foi modelada como ponto de extensão para metadados de divulgação trimestral.
