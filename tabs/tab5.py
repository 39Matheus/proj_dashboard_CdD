
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict

def render_tab5(df_1, data_inicial, data_final):
    st.subheader("📋 Dados filtrados")
    st.dataframe(df_1)