import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import re
from html import unescape
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import os
import gdown

#Funções

# Função multilabel binarizer (Tab4)
def multilabel_binarize(df, column):
    mlb = MultiLabelBinarizer()
    # Preenche NaN com listas vazias para evitar erros
    col_data = df[column].apply(lambda x: x if isinstance(x, list) else [])
    dummies = pd.DataFrame(mlb.fit_transform(col_data), columns=mlb.classes_, index=df.index)
    return dummies

# Função para contar categorias, gêneros ou idiomas
def count_items(df, col):
    counts = defaultdict(int)
    for row in df[col]:
        if isinstance(row, list):
            for item in row:
                if isinstance(item, str):
                    counts[item.lower()] += 1
    return pd.DataFrame(counts.items(), columns=[col.capitalize(), 'Count']).sort_values('Count', ascending=False)

#Normalizar Tags
def normalize_tags(x):
    if isinstance(x, dict):
        return list(x.keys())
    elif isinstance(x, str):
        try:
            import ast
            parsed = ast.literal_eval(x)
            if isinstance(parsed, dict):
                return list(parsed.keys())
            elif isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []


# Função para normalizar a coluna supported_languages
def normalize_languages(entry):
    if isinstance(entry, list):
        raw = entry
    elif isinstance(entry, str):
        raw = entry.split(',')
    else:
        return []

    langs = set()
    for item in raw:
        item = unescape(item)
        item = re.sub(r'\[/?b\]|<.*?>|&lt;.*?&gt;', '', item)
        item = item.replace(';', '').strip()
        for sub_lang in re.split(r'[,\n]+', item):
            sub_lang = sub_lang.strip()
            if sub_lang and '#' not in sub_lang and '/' not in sub_lang:
                langs.add(normalization_map.get(sub_lang, sub_lang))
    return list(langs)


# Caminhos dos arquivos
original_path = "games.json"
reduced_path = "games_reduzido.json.gz"

# Verifica se o arquivo original existe, se não, baixa
if not os.path.exists(original_path):
    file_id = "1uF1nhyZ7ghk9flT2uuCgTRz70gCMpyx0"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, original_path, quiet=False)

# Verifica se o arquivo reduzido já existe
if os.path.exists(reduced_path):
    df_1 = pd.read_json(reduced_path, orient="records", lines=True, compression="gzip")
else:
    with st.spinner("Processando JSON..."):
        raw = pd.read_json(original_path).transpose().rename_axis("AppID").reset_index()
        filtro_col = [
            'name', 'release_date', 'price', 'dlc_count', 'windows', 'mac', 'linux',
            'achievements', 'supported_languages', 'developers', 'publishers',
            'categories', 'genres', 'positive', 'negative', 'estimated_owners', 'tags'
        ]
        reduzido = raw[filtro_col]
        reduzido.to_json(reduced_path, orient="records", lines=True, compression="gzip")
        st.success("Arquivo processado com sucesso!")

    df_1 = pd.read_json(reduced_path, orient="records", lines=True, compression="gzip")

# Cria coluna de reviews se ainda não existir
if 'reviews' not in df_1.columns:
    df_1.insert(df_1.columns.get_loc('negative') + 1, 'reviews', df_1['positive'] + df_1['negative'])

# --------------------------
# Normalização de dados
# --------------------------

# Mapeamento para normalização de línguas
normalization_map = {
    "Slovakian": "Slovak",
    "English (full audio)": "English",
    "Japanese (all with full audio support)": "Japanese",
    "Traditional Chinese (text only)": "Traditional Chinese",
    "English Dutch  English": "English",
    "Portuguese - Portugal": "Portuguese",
}

df_1['supported_languages'] = df_1['supported_languages'].apply(normalize_languages)

print(f"Tags pré norm: {df_1['tags'].head(10)}")
df_1['tags'] = df_1['tags'].apply(normalize_tags)
print(f"Tags pós norm: {df_1['tags'].head(10)}")

# DEBUG: Verificando se há problemas com as tags
print("Entradas nulas ou inválidas em tags:", df_1['tags'].isna().sum())
print("Entradas vazias em tags:", (df_1['tags'].apply(lambda x: len(x) == 0)).sum())

# --------------------------
# Processamentos adicionais
# --------------------------

# Converte release_date para datetime e remove datas inválidas
df_1['release_date'] = pd.to_datetime(df_1['release_date'], errors='coerce')
df_1 = df_1[df_1['release_date'].notna()]

# Contagens principais
genre_df = count_items(df_1, 'genres').head(12)
cat_df = count_items(df_1, 'categories').head(12)

# Contagem de idiomas
lang_counts = defaultdict(int)
for langs in df_1['supported_languages']:
    for lang in langs:
        lang_counts[lang] += 1
lang_df = pd.DataFrame(lang_counts.items(), columns=['Language', 'Count']).sort_values('Count', ascending=False).head(12)

#Sidebar

cutoff_date = pd.Timestamp('2024-08-31')
with st.sidebar:
    st.markdown("### 📆 Filtro de Período de Visualização (com mês)")
    data_min = pd.Timestamp("2006-01-01")
    data_max = min(cutoff_date, pd.to_datetime(df_1['release_date'], errors='coerce').max())

    data_inicial = st.date_input("Data inicial", value=data_min, min_value=data_min, max_value=data_max)
    data_final = st.date_input("Data final", value=data_max, min_value=data_inicial, max_value=data_max)

    data_inicial = pd.Timestamp(data_inicial)
    data_final = pd.Timestamp(data_final)

# Aplica filtro de data no DataFrame
df_1 = df_1[(df_1['release_date'] >= data_inicial) & (df_1['release_date'] <= data_final)]

# Interface em abas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Gráficos 1", "📈 Gráficos 2", "⌛ Gráficos Temporais" , "🧠 Análises Avançadas", "📁 Tabela Completa"])

with tab1:
    st.subheader("Visão Geral")

    # Filtragem do DataFrame com base nas datas escolhidas
    df_temporal = df_1.copy()
    df_temporal['release_date'] = pd.to_datetime(df_temporal['release_date'], errors='coerce')
    df_temporal = df_temporal[
        (df_temporal['release_date'] >= data_inicial) &
        (df_temporal['release_date'] <= data_final)
    ].reset_index(drop=True)

    # Recontagens para o período
    genre_df_temp = count_items(df_temporal, 'genres').head(12)
    cat_df_temp = count_items(df_temporal, 'categories').head(12)

    lang_counts_temp = defaultdict(int)
    for langs in df_temporal['supported_languages']:
        for lang in langs:
            lang_counts_temp[lang] += 1
    lang_df_temp = pd.DataFrame(lang_counts_temp.items(), columns=['Language', 'Count']).sort_values('Count', ascending=False).head(12)

    tag_counts_temp = defaultdict(int)
    for entry in df_temporal['tags']:
        if isinstance(entry, list) and all(isinstance(tag, str) for tag in entry):
            for tag in entry:
                tag_counts_temp[tag] += 1

    tag_df_temp = pd.DataFrame(tag_counts_temp.items(), columns=['Tags', 'Count']).sort_values('Count', ascending=False).head(12)

    # Gráficos
    cols = st.columns(2)

    with cols[0]:
        st.subheader("Top 12 Gêneros")
        st.plotly_chart(px.bar(genre_df_temp, x="Genres", y="Count", color="Genres"), use_container_width=True)

    with cols[1]:
        st.subheader("Top 12 Categorias")
        st.plotly_chart(px.bar(cat_df_temp, x="Categories", y="Count", color="Categories"), use_container_width=True)

    cols = st.columns(2)

    with cols[0]:
        st.subheader("Top 12 Idiomas")
        st.plotly_chart(px.bar(lang_df_temp, x="Language", y="Count", color="Language"), use_container_width=True)

    with cols[1]:
        st.subheader("Top 12 Tags")
        st.plotly_chart(px.bar(tag_df_temp, x="Tags", y="Count", color="Tags"), use_container_width=True)


with tab2:
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

with tab3:
    # -- 1. Converte release_date para datetime
    release_dates = pd.to_datetime(df_1['release_date'], errors='coerce')

    # -- 2. Agrupa por ano-mês
    release_ym = release_dates.dt.to_period('M').dt.to_timestamp()
    lancamentos_por_mes = release_ym.value_counts().sort_index()
    df_lancamentos = lancamentos_por_mes.reset_index()
    df_lancamentos.columns = ['AnoMes', 'Quantidade']

    # -- 3. Define datas de corte
    cutoff_date = pd.Timestamp('2024-08-31')
    base_date = pd.Timestamp('2006-01-01')
    base_ordinal = base_date.toordinal()

    # -- 4. Sidebar para filtro de datas com precisão mensal
    ### Movida para ser global ###

    # -- 5. Filtragem visual
    filtro_visual = df_lancamentos[
        (df_lancamentos['AnoMes'] >= data_inicial) &
        (df_lancamentos['AnoMes'] <= data_final)
    ]

    # -- 6. Dados reais para regressão (até agosto/2024)
    df_lancamentos_filtrado = df_lancamentos[df_lancamentos['AnoMes'] <= cutoff_date]

    # -- 7. Regressão polinomial grau 3
    x_train = df_lancamentos_filtrado['AnoMes'].map(lambda d: d.toordinal() - base_ordinal).values.reshape(-1, 1)
    y_train = df_lancamentos_filtrado['Quantidade'].values

    poly = PolynomialFeatures(degree=3)
    x_train_poly = poly.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # -- 8. Projeção futura (12 meses após ago/2024)
    future_dates = [cutoff_date + pd.DateOffset(months=i) for i in range(1, 13)]
    x_future = np.array([(d.toordinal() - base_ordinal) for d in future_dates]).reshape(-1, 1)
    x_future_poly = poly.transform(x_future)
    y_future = model.predict(x_future_poly)

    # -- 9. Linha polinomial (real + futuro)
    x_all = np.concatenate([x_train, x_future])
    x_all_poly = poly.transform(x_all)
    y_all = model.predict(x_all_poly)
    dates_all = list(df_lancamentos_filtrado['AnoMes']) + future_dates

    # -- 10. Gráfico interativo
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtro_visual['AnoMes'],
        y=filtro_visual['Quantidade'],
        mode='lines+markers',
        name='Dados Reais (Filtrados)',
        line=dict(color='crimson'),
        marker=dict(symbol='circle', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=dates_all[:len(x_train)],
        y=y_all[:len(x_train)],
        mode='lines',
        name='Tendência Polinomial (Grau 3)',
        line=dict(color='navy', dash='solid')
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_future,
        mode='lines',
        name='Projeção Futura',
        line=dict(color='navy', dash='dash')
    ))

    # -- Fórmula do polinômio
    coef = model.coef_
    intercept = model.intercept_
    equation = (
        f"y = {intercept:.2f} + "
        f"{coef[1]:.2e}·x + "
        f"{coef[2]:.2e}·x² + "
        f"{coef[3]:.2e}·x³"
    )

    fig.add_annotation(
        text=f"<b>Equação (x = anos desde 2006):</b><br>{equation}",
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor='left', yanchor='top',
        showarrow=False,
        align="right",
        font=dict(size=12, color="gray"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgray",
        borderwidth=1
    )

    fig.update_layout(
        title='Quantidade de Jogos Lançados (com Projeção Polinomial)',
        xaxis_title='Ano',
        yaxis_title='Número de Jogos',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    fig.update_xaxes(
        dtick="M12",
        tickformat="%Y",
        tickangle=45,
        range=[data_inicial, future_dates[-1]]
    )
    fig.update_yaxes(rangemode='tozero')

    st.plotly_chart(fig, use_container_width=True)

    # -- 11. Preço médio mensal com desvio padrão
    df_preco = df_1.copy()
    df_preco['release_date'] = pd.to_datetime(df_preco['release_date'], errors='coerce')
    df_preco = df_preco[df_preco['release_date'].notna()]
    df_preco['AnoMes'] = df_preco['release_date'].dt.to_period('M').dt.to_timestamp()

    # -- Agrupa preço médio e desvio padrão por mês
    preco_stats = df_preco.groupby('AnoMes')['price'].agg(['mean', 'std']).reset_index()
    preco_stats.columns = ['AnoMes', 'Preço Médio', 'Desvio Padrão']

    # -- Filtra visualização conforme janela selecionada
    preco_stats_filtrado = preco_stats[
        (preco_stats['AnoMes'] >= data_inicial) & 
        (preco_stats['AnoMes'] <= data_final)
    ].copy()

    # -- Cria faixas de ±1 desvio
    preco_stats_filtrado['Faixa Superior'] = preco_stats_filtrado['Preço Médio'] + preco_stats_filtrado['Desvio Padrão']
    preco_stats_filtrado['Faixa Inferior'] = preco_stats_filtrado['Preço Médio'] - preco_stats_filtrado['Desvio Padrão']

    # -- Cria gráfico com linha central + sombra (desvio padrão)
    fig_preco = go.Figure()

    # Faixa de desvio padrão (sombras)
    fig_preco.add_traces([
        go.Scatter(
            x=preco_stats_filtrado['AnoMes'],
            y=preco_stats_filtrado['Faixa Superior'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name="+1σ",
        ),
        go.Scatter(
            x=preco_stats_filtrado['AnoMes'],
            y=preco_stats_filtrado['Faixa Inferior'],
            fill='tonexty',
            fillcolor='rgba(1, 35, 79, 0.15)',
            mode='lines',
            line=dict(width=0),
            name="±1 Desvio Padrão",
            hoverinfo='skip',
        )
    ])

    # Linha do preço médio
    fig_preco.add_trace(go.Scatter(
        x=preco_stats_filtrado['AnoMes'],
        y=preco_stats_filtrado['Preço Médio'],
        mode='lines+markers',
        name='Preço Médio',
        line=dict(color='#01234f', width=2),
        marker=dict(size=5),
        hovertemplate='Mês: %{x|%b/%Y}<br>Preço Médio: R$ %{y:.2f}<extra></extra>'
    ))

    # Layout
    fig_preco.update_layout(
        title="📈 Preço Médio Mensal dos Jogos (+ Desvio Padrão)",
        xaxis_title="Ano",
        yaxis_title="Preço (R$)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )

    fig_preco.update_xaxes(
        dtick="M12",
        tickformat="%Y",
        tickangle=45,
        range=[data_inicial, data_final]
    )

    fig_preco.update_yaxes(rangemode='tozero')

    # Mostra o gráfico
    st.plotly_chart(fig_preco, use_container_width=True)



with tab4:
    st.subheader("🔍 Análises Avançadas: Correlações Multilabel")

    # Usa data_inicial e data_final do filtro do tab3 para filtrar df_1
    df_filtered = df_1.copy()
    df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'], errors='coerce')
    df_filtered = df_filtered[
        (df_filtered['release_date'] >= data_inicial) &
        (df_filtered['release_date'] <= data_final)
    ].reset_index(drop=True)

    st.markdown(f"Jogos no intervalo: {len(df_filtered)}")



    # Binariza colunas multilabel
    df_genres = multilabel_binarize(df_filtered, 'genres')
    df_categories = multilabel_binarize(df_filtered, 'categories')
    df_languages = multilabel_binarize(df_filtered, 'supported_languages')
    
    df_tags = multilabel_binarize(df_filtered, 'tags')

    # Mapeamento para seleção do usuário
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
    else:
        # Correlação entre os grupos (colunas)
        corr = pd.DataFrame(np.corrcoef(X.T, Y.T)[:X.shape[1], X.shape[1]:], index=X.columns, columns=Y.columns)

        # Filtra correlações com valor absoluto maior que um limiar para melhor visualização
        threshold = 0.15
        corr_filtered = corr.where(np.abs(corr) >= threshold)

        if corr_filtered.isnull().all().all():
            st.info("Nenhuma correlação acima do limiar encontrada.")
        else:
            # Substitui NaN por 0 para heatmap
            heatmap_data = corr_filtered.fillna(0)

            # Heatmap com Plotly Express
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlação"),
                aspect="auto",
                text_auto=".2f",
                title=f"Heatmap de Correlação: {escolha} (|corr| >= {threshold})"
            )

            fig.update_layout(
                height=600,
                margin=dict(l=100, r=100, t=80, b=80),
                coloraxis_colorbar=dict(
                    title="Correlação",
                    tickvals=[-1, 0, 1],
                    ticktext=["-1 (Negativa)", "0 (Nenhuma)", "1 (Positiva)"]
                )
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
    with col1:
        grupo1_nome = st.selectbox("Primeiro grupo:", sorted(grupos.keys()), key="grupo1")
    with col2:
        grupo2_nome = st.selectbox("Segundo grupo:", sorted(grupos.keys()), key="grupo2")

    grupo1_df = grupos[grupo1_nome]
    grupo2_df = grupos[grupo2_nome]

    item1 = st.selectbox(f"Selecione um item de {grupo1_nome}:", sorted(grupo1_df.columns), key="item1")
    item2 = st.selectbox(f"Selecione um item de {grupo2_nome}:", sorted(grupo2_df.columns), key="item2")

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
        else:
            return 'Nenhum'

    df_plot['classe'] = df_plot.apply(classifica, axis=1)
    df_plot = df_plot[df_plot['classe'] != 'Nenhum']

    cor_map = {
        f'Apenas {item1}': 'blue',
        f'Apenas {item2}': 'red',
        'Ambos': 'green'
    }

    # Y fixo, um nível para cada categoria
    categorias = [f'Apenas {item1}', f'Apenas {item2}', 'Ambos']
    pos_y = {cat: i for i, cat in enumerate(categorias[::-1], start=1)}  # Inverte para "Ambos" em cima

    fig = go.Figure()

    for classe, cor in cor_map.items():
        if classe in categorias:
            df_sub = df_plot[df_plot['classe'] == classe].sort_values('release_date')
            if df_sub.empty:
                continue  # Pula se não tem dados
            y_values = [pos_y[classe]] * len(df_sub)
            fig.add_trace(go.Scatter(
                x=df_sub['release_date'],
                y=y_values,
                mode='markers',
                name=classe,
                line=dict(color=cor, shape='spline'),
                marker=dict(color=cor, size=8, opacity=0.7, line=dict(width=0)),
                hovertemplate='<b>%{text}</b><br>Data: %{x|%d %b %Y}<extra></extra>',
                text=df_sub['name']
            ))

    fig.update_yaxes(
        tickmode='array',
        tickvals=list(pos_y.values()),
        ticktext=list(pos_y.keys()),
        title_text='Categoria',
        autorange='reversed'
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
    
with tab5:
    st.subheader("📋 Dados filtrados")
    st.dataframe(df_1)