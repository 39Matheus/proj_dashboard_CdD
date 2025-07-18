import streamlit as st
from data_loader import download_and_prepare_data
from tabs import tab1, tab2, tab3, tab4, tab5
import pandas as pd

st.set_page_config(layout="wide", page_title="Steam Dashboard")
st.title("🎮 Steam Dashboard")

# Carrega os dados uma vez com cache
df_1 = download_and_prepare_data()

# Sidebar de filtro de datas
cutoff_date = pd.Timestamp('2024-08-31')
with st.sidebar:
    st.markdown("### 📆 Filtro de Período de Visualização (com mês)")
    data_min = pd.Timestamp("2006-01-01")
    data_max = min(cutoff_date, pd.to_datetime(df_1['release_date'], errors='coerce').max())

    data_inicial = st.date_input("Data inicial", value=data_min, min_value=data_min, max_value=data_max)
    data_final = st.date_input("Data final", value=data_max, min_value=data_inicial, max_value=data_max)

    data_inicial = pd.Timestamp(data_inicial)
    data_final = pd.Timestamp(data_final)

# Aplica o filtro temporal no DataFrame
df_1 = df_1[(df_1['release_date'] >= data_inicial) & (df_1['release_date'] <= data_final)]

# Layout com tabs
tabs = st.tabs(["📊 Gráficos 1", "📈 Gráficos 2", "⌛ Gráficos Temporais" , "🧠 Análises Avançadas", "📁 Tabela Completa"])

with tabs[0]:
    tab1.render_tab1(df_1, data_inicial, data_final)
with tabs[1]:
    tab2.render_tab2(df_1, data_inicial, data_final)
with tabs[2]:
    tab3.render_tab3(df_1, data_inicial, data_final)
with tabs[3]:
    tab4.render_tab4(df_1, data_inicial, data_final)
with tabs[4]:
    tab5.render_tab5(df_1, data_inicial, data_final)