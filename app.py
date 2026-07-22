"""Streamlit dashboard for the quantitative Brazilian value investing system."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from config import settings
from modules.database import SQLiteRepository
from modules.logging_config import configure_logging
from modules.pipeline import run_pipeline

st.set_page_config(page_title="Value Investing Quant Brasil", layout="wide")
configure_logging(settings.log_dir)
st.title("Sistema Quantitativo de Value Investing para Ações Brasileiras")
st.caption("Coleta dados, calcula fatores, aplica filtros, monta carteira e acompanha backtest.")

if st.sidebar.button("Atualizar dados e carteira", type="primary"):
    with st.spinner("Executando pipeline..."):
        st.session_state["result"] = run_pipeline(settings)

repo = SQLiteRepository(settings.database_path)
indicators = repo.read_table("indicators")
portfolio = repo.read_table("portfolio")

summary, ranking_tab, portfolio_tab, backtest_tab, excluded_tab = st.tabs(["Dashboard", "Ranking", "Carteira Atual", "Backtest", "Empresas Excluídas"])
with summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("Empresas avaliadas", len(indicators["ticker"].unique()) if not indicators.empty else 0)
    col2.metric("Ativos na carteira", len(portfolio["ticker"].unique()) if not portfolio.empty else 0)
    col3.metric("Score máximo", f"{indicators['score'].max():.2f}" if not indicators.empty else "-")
with ranking_tab:
    st.subheader("Ranking e fatores")
    st.dataframe(indicators.sort_values("score", ascending=False) if not indicators.empty else indicators, use_container_width=True)
with portfolio_tab:
    st.subheader("Carteira atual")
    st.dataframe(portfolio, use_container_width=True)
    if not portfolio.empty:
        fig = px.pie(portfolio, names="ticker", values="weight", title="Pesos da carteira")
        st.plotly_chart(fig, use_container_width=True)
with backtest_tab:
    st.subheader("Backtest")
    result = st.session_state.get("result")
    if result and not result["backtest"].empty:
        bt = result["backtest"]
        st.plotly_chart(px.line(bt, x="date", y="equity", title="Evolução do patrimônio"), use_container_width=True)
        dd = bt["equity"] / bt["equity"].cummax() - 1
        st.plotly_chart(px.area(x=bt["date"], y=dd, title="Drawdown"), use_container_width=True)
        st.json(result["metrics"])
    else:
        st.info("Execute a atualização para gerar o backtest da carteira atual.")
with excluded_tab:
    st.subheader("Empresas excluídas e motivos")
    excluded = indicators[indicators["exclusion_reason"].fillna("").ne("")] if not indicators.empty else indicators
    st.dataframe(excluded[["ticker", "exclusion_reason"]] if not excluded.empty else excluded, use_container_width=True)
