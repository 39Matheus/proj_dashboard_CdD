import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
from data_loader import count_items

def render_tab1(df_1, data_inicial, data_final):

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
