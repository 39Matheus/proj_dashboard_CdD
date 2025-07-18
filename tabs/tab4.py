import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import multilabel_binarize, normalize_tags

def render_tab4(df_1, data_inicial, data_final):
    st.subheader("🔍 Análises Avançadas: Correlações Multilabel")

    # Filtra dados conforme data
    df_filtered = df_1.copy()
    df_filtered = df_filtered[(df_filtered['release_date'] >= data_inicial) & (df_filtered['release_date'] <= data_final)]
    st.markdown(f"Jogos no intervalo: {len(df_filtered)}")

    # Binariza colunas multilabel
    df_genres = multilabel_binarize(df_filtered, 'genres')
    df_categories = multilabel_binarize(df_filtered, 'categories')
    df_languages = multilabel_binarize(df_filtered, 'supported_languages')
    df_filtered['tags_list'] = df_filtered['tags']
    df_tags = multilabel_binarize(df_filtered, 'tags_list')

    options = {
        "Gêneros x Línguas": (df_genres, df_languages),
        "Gêneros x Categorias": (df_genres, df_categories),
        "Línguas x Categorias": (df_languages, df_categories),
        "Gêneros x Tags": (df_genres, df_tags),
        "Línguas x Tags": (df_languages, df_tags),
        "Categorias x Tags": (df_categories, df_tags)
    }

    escolha = st.selectbox("Selecione a correlação a ser exibida:", list(options.keys()))
    X, Y = options[escolha]

    if X.shape[0] != Y.shape[0]:
        st.warning("Erro: número de linhas diferente entre as categorias selecionadas, verifique os dados.")
        return

    corr = pd.DataFrame(np.corrcoef(X.T, Y.T)[:X.shape[1], X.shape[1]:], index=X.columns, columns=Y.columns)
    threshold = 0.15
    corr_filtered = corr.where(np.abs(corr) >= threshold)

    if corr_filtered.isnull().all().all():
        st.info("Nenhuma correlação acima do limiar encontrada.")
    else:
        fig = px.imshow(
            corr_filtered.fillna(0),
            color_continuous_scale='RdBu', zmin=-1, zmax=1,
            labels=dict(color="Correlação"), aspect="auto", text_auto=".2f",
            title=f"Heatmap de Correlação: {escolha} (|corr| >= {threshold})"
        )
        fig.update_layout(
            height=600,
            margin=dict(l=100, r=100, t=80, b=80),
            coloraxis_colorbar=dict(title="Correlação", tickvals=[-1, 0, 1], ticktext=["-1", "0", "1"]),
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Interseção de Duas Categorias ao Longo do Tempo")
    grupos = {
        "Categorias": df_categories,
        "Gêneros": df_genres,
        "Línguas": df_languages,
        "Tags": df_tags
    }

    col1, col2 = st.columns(2)
    grupo1_nome = col1.selectbox("Primeiro grupo:", sorted(grupos.keys()), key="grupo1")
    grupo2_nome = col2.selectbox("Segundo grupo:", sorted(grupos.keys()), key="grupo2")

    grupo1_df = grupos[grupo1_nome]
    grupo2_df = grupos[grupo2_nome]

    item1 = st.selectbox(f"Item de {grupo1_nome}:", sorted(grupo1_df.columns), key="item1")
    item2 = st.selectbox(f"Item de {grupo2_nome}:", sorted(grupo2_df.columns), key="item2")

    df_plot = df_filtered[['release_date', 'name']].copy()
    df_plot['possui_1'] = grupo1_df[item1]
    df_plot['possui_2'] = grupo2_df[item2]

    def classifica(row):
        if row['possui_1'] and row['possui_2']:
            return 'Ambos'
        elif row['possui_1']:
            return f'Apenas {item1}'
        elif row['possui_2']:
            return f'Apenas {item2}'
        return 'Nenhum'

    df_plot['classe'] = df_plot.apply(classifica, axis=1)
    df_plot = df_plot[df_plot['classe'] != 'Nenhum']

    cor_map = {
        f'Apenas {item1}': 'blue',
        f'Apenas {item2}': 'red',
        'Ambos': 'green'
    }

    categorias = [f'Apenas {item1}', f'Apenas {item2}', 'Ambos']
    pos_y = {cat: i for i, cat in enumerate(categorias[::-1], start=1)}

    fig = go.Figure()
    for classe, cor in cor_map.items():
        if classe in df_plot['classe'].values:
            df_sub = df_plot[df_plot['classe'] == classe].sort_values('release_date')
            fig.add_trace(go.Scatter(
                x=df_sub['release_date'], y=[pos_y[classe]] * len(df_sub),
                mode='markers', name=classe,
                marker=dict(color=cor, size=8, opacity=0.7),
                hovertemplate='<b>%{text}</b><br>Data: %{x|%d %b %Y}<extra></extra>',
                text=df_sub['name']
            ))

    fig.update_yaxes(
        tickmode='array', tickvals=list(pos_y.values()),
        ticktext=list(pos_y.keys()), title_text='Categoria', autorange='reversed'
    )
    fig.update_layout(
        height=400,
        xaxis_title="Data de Lançamento",
        template="plotly_white",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
