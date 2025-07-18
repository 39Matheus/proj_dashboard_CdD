import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
#from data_loader import count_items, normalize_tags

def render_tab2(df_1, data_inicial, data_final):
# Filtra o df_1 pela faixa temporal selecionada
    df_filtered = df_1[
        (df_1['release_date'] >= data_inicial) &
        (df_1['release_date'] <= data_final)
    ]

    cols = st.columns(2)

    # Jogos por Sistema Operacional
    with cols[0]:
        os_df = df_filtered[['windows', 'mac', 'linux']].sum().reset_index()
        os_df.columns = ['Sistema Operacional', 'Número de Jogos']
        st.subheader("Sistemas Operacionais")
        cores_OS = {
            'windows': '#01234f',
            'mac': "#868686",
            'linux': "#0F0303"
        }
        st.plotly_chart(
            px.bar(
                os_df, x="Sistema Operacional", y="Número de Jogos", 
                color="Sistema Operacional", color_discrete_map=cores_OS
            ), 
            use_container_width=True
        )

    # Lançamentos por Mês
    with cols[1]:
        st.subheader("Lançamentos por Mês")
        release_dates = pd.to_datetime(df_filtered['release_date'], errors='coerce')
        release_month_numbers = release_dates.dt.month
        meses_pt = {
            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
        }
        lanc_por_mes = release_month_numbers.value_counts().sort_index()
        df_lanc = pd.DataFrame({
            "Mês": lanc_por_mes.index.map(meses_pt),
            "Quantidade": lanc_por_mes.values
        })
        st.plotly_chart(px.bar(df_lanc, x="Mês", y="Quantidade", color="Mês"), use_container_width=True)

    # Aprovação (sem criar coluna no DataFrame)
    df_analises = df_filtered[df_filtered['reviews'] >= 10].copy()

    approval_series = df_analises['positive'] / (df_analises['positive'] + df_analises['negative'])
    approval_series = approval_series.dropna()

    mean_approval = approval_series.mean()
    std_approval = approval_series.std()

    df_temp = pd.DataFrame({'approval': approval_series})

    fig = px.histogram(
        df_temp,
        x='approval',
        nbins=30,
        histnorm='probability density',
        marginal='violin',
        opacity=0.75,
        color_discrete_sequence=['#01234f']
    )

    fig.add_vline(
        x=mean_approval,
        line_dash='dash',
        line_color='red',
        annotation_text=f"Média: {mean_approval:.2f}",
        annotation_position="top right"
    )

    fig.add_vrect(
        x0=mean_approval - std_approval,
        x1=mean_approval + std_approval,
        fillcolor='red',
        opacity=0.2,
        layer='below',
        line_width=0,
        annotation_text="±1 Desvio",
        annotation_position="top left"
    )

    fig.update_layout(
        title='Distribuição da taxa de aprovação dos jogos (mínimo de 10 avaliações)',
        xaxis_title='Taxa de Aprovação (positive / reviews)',
        yaxis_title='Densidade de Jogos',
        template='plotly_white',
        bargap=0.05
    )

    fig.update_traces(marker_line_width=0.5)

    st.plotly_chart(fig, use_container_width=True)