import pandas as pd
import numpy as np
import ast
import re
from html import unescape
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
import streamlit as st
import os
import gdown

normalization_map = {
    'english': 'English',
    'brazilian portuguese': 'Portuguese - Brazil',
    'schinese': 'Chinese - Simplified',
    'tchinese': 'Chinese - Traditional',
    'latam': 'Spanish - Latin America',
    'spanish (latin america)': 'Spanish - Latin America',
    'spanish - latin america': 'Spanish - Latin America',
    'japanese': 'Japanese',
    'french': 'French',
    'german': 'German',
    'korean': 'Korean',
    'russian': 'Russian',
    'italian': 'Italian',
    'portuguese': 'Portuguese',
    'brazilian': 'Portuguese - Brazil',
    'polish': 'Polish',
    'turkish': 'Turkish'
}

# Função para contar categorias, gêneros ou idiomas
def count_items(df, col):
    counts = defaultdict(int)
    for row in df[col]:
        if isinstance(row, list):
            for item in row:
                if isinstance(item, str):
                    counts[item.lower()] += 1
    return pd.DataFrame(counts.items(), columns=[col.capitalize(), 'Count']).sort_values('Count', ascending=False)

@st.cache_data(show_spinner="Carregando dados...")
def download_and_prepare_data(original_path="games.json", reduced_path="games_reduzido.json.gz"):
    if not os.path.exists(original_path):
        file_id = "1uF1nhyZ7ghk9flT2uuCgTRz70gCMpyx0"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, original_path, quiet=False)

    if os.path.exists(reduced_path):
        df = pd.read_json(reduced_path, orient="records", lines=True, compression="gzip")
    else:
        raw = pd.read_json(original_path).transpose().rename_axis("AppID").reset_index()
        filtro_col = [
            'name', 'release_date', 'price', 'dlc_count', 'windows', 'mac', 'linux',
            'achievements', 'supported_languages', 'developers', 'publishers',
            'categories', 'genres', 'positive', 'negative', 'estimated_owners', 'tags'
        ]
        reduzido = raw[filtro_col]
        reduzido.to_json(reduced_path, orient="records", lines=True, compression="gzip")
        df = reduzido

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['tags'] = df['tags'].apply(normalize_tags)
    df['supported_languages'] = df['supported_languages'].apply(normalize_languages)
    if 'reviews' not in df.columns:
        df.insert(df.columns.get_loc('negative') + 1, 'reviews', df['positive'] + df['negative'])

    return df

def normalize_tags(entry):
    if not isinstance(entry, (dict, str)):
        return []
    try:
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        if isinstance(entry, dict):
            return list(entry.keys())
        return []
    except Exception:
        return []


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
        for sub_lang in re.split(r'[\,\n]+', item):
            sub_lang = sub_lang.strip()
            if sub_lang and '#' not in sub_lang and '/' not in sub_lang:
                langs.add(normalization_map.get(sub_lang.lower(), sub_lang))
    return list(langs)


def multilabel_binarize(df, column):
    mlb = MultiLabelBinarizer()
    col_data = df[column].apply(lambda x: x if isinstance(x, list) else [])
    dummies = pd.DataFrame(mlb.fit_transform(col_data), columns=mlb.classes_, index=df.index)
    return dummies