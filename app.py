import gdown
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict
import json
import re
from html import unescape
from sklearn.preprocessing import MultiLabelBinarizer

file_id = "1uF1nhyZ7ghk9flT2uuCgTRz70gCMpyx0"
url = f"https://drive.google.com/uc?id={file_id}"


st.set_page_config(layout="wide", page_title="Análise de Jogos Steam")
st.title("🎮 Dashboard de Análise de Jogos da Steam")

output = "games.json"
gdown.download(url, output, quiet=False)

DATA = pd.read_json(output).transpose().rename_axis('AppID').reset_index()
# Seleção de colunas úteis
filtro_col = ['name', 'release_date', 'price', 'dlc_count', 'windows', 'mac', 'linux',
              'achievements', 'supported_languages', 'developers', 'publishers',
              'categories', 'genres', 'positive', 'negative', 'estimated_owners', 'tags']
df_1 = DATA[filtro_col].copy()

# Cria coluna de reviews se ainda não existir
if 'reviews' not in df_1.columns:
    df_1.insert(df_1.columns.get_loc('negative') + 1, 'reviews', df_1['positive'] + df_1['negative'])

# Normalização de idiomas
normalization_map = {
    "Slovakian": "Slovak",
    "English (full audio)": "English",
    "Japanese (all with full audio support)": "Japanese",
    "Traditional Chinese (text only)": "Traditional Chinese",
    "English Dutch  English": "English",
    "Portuguese - Portugal": "Portuguese",
}

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

df_1['supported_languages'] = df_1['supported_languages'].apply(normalize_languages)

# Função para contar categorias/gêneros/etc
def count_items(df, col):
    counts = defaultdict(int)
    for row in df[col]:
        if isinstance(row, list):
            for item in row:
                counts[item.lower()] += 1
    return pd.DataFrame(counts.items(), columns=[col.capitalize(), 'Count']).sort_values('Count', ascending=False)

# Dados para os gráficos principais
genre_df = count_items(df_1, 'genres').head(12)
cat_df = count_items(df_1, 'categories').head(12)

lang_counts = defaultdict(int)
for langs in df_1['supported_languages']:
    for lang in langs:
        lang_counts[lang] += 1
lang_df = pd.DataFrame(lang_counts.items(), columns=['Language', 'Count']).sort_values('Count', ascending=False).head(12)

# Tabs principais
tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "📁 Tabela Completa", "🔬 Análises Avançadas"])

with tab1:
    st.subheader("Top 12 Gêneros")
    st.plotly_chart(px.bar(genre_df, x="Genres", y="Count", color="Genres"), use_container_width=True)

    st.subheader("Top 12 Categorias")
    st.plotly_chart(px.bar(cat_df, x="Categories", y="Count", color="Categories"), use_container_width=True)

    st.subheader("Top 12 Idiomas")
    st.plotly_chart(px.bar(lang_df, x="Language", y="Count", color="Language"), use_container_width=True)

with tab2:
    st.subheader("📋 Dados filtrados")
    st.dataframe(df_1)

with tab3:
    st.subheader("🔍 Correlações (exemplo básico com gênero x linguagem)")

    try:
        df_genres = MultiLabelBinarizer().fit_transform(df_1['genres'].dropna())
        df_langs = MultiLabelBinarizer().fit_transform(df_1['supported_languages'].dropna())

        if df_genres.shape[0] == df_langs.shape[0]:
            corr_matrix = np.corrcoef(df_genres.T @ df_langs)
            st.write("⚠️ Heatmap omitido para evitar poluição visual. Ideal usar Plotly Heatmap interativo filtrável.")
        else:
            st.warning("Número de jogos com gêneros e idiomas não está compatível. Corrija ou filtre.")
    except Exception as e:
        st.error(f"Erro ao calcular correlação: {e}")
