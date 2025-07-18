import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

def render_tab3(df_1, data_inicial, data_final):
    release_dates = pd.to_datetime(df_1['release_date'], errors='coerce')
    release_ym = release_dates.dt.to_period('M').dt.to_timestamp()
    lancamentos_por_mes = release_ym.value_counts().sort_index()
    df_lancamentos = lancamentos_por_mes.reset_index()
    df_lancamentos.columns = ['AnoMes', 'Quantidade']

    cutoff_date = data_final
    base_date = pd.Timestamp('2006-01-01')
    base_ordinal = base_date.toordinal()

    filtro_visual = df_lancamentos[
        (df_lancamentos['AnoMes'] >= data_inicial) & 
        (df_lancamentos['AnoMes'] <= data_final)
    ]

    df_lancamentos_filtrado = df_lancamentos[
        (df_lancamentos['AnoMes'] >= data_inicial) & 
        (df_lancamentos['AnoMes'] <= cutoff_date)
    ]

    x_train = df_lancamentos_filtrado['AnoMes'].map(lambda d: d.toordinal() - base_ordinal).values.reshape(-1, 1)
    y_train = df_lancamentos_filtrado['Quantidade'].values

    future_dates = [cutoff_date + pd.DateOffset(months=i) for i in range(1, 13)]
    x_future = np.array([(d.toordinal() - base_ordinal) for d in future_dates]).reshape(-1, 1)

    fig = go.Figure()

    # Dados reais filtrados
    fig.add_trace(go.Scatter(
        x=filtro_visual['AnoMes'],
        y=filtro_visual['Quantidade'],
        mode='lines+markers',
        name='Dados Reais (Filtrados)',
        line=dict(color='crimson'),
        marker=dict(symbol='circle', size=6)
    ))

    cores = ['green', 'orange', 'navy', 'purple']
    graus_lista = [1, 2, 3, 4]

    annotations = []

    # Vamos guardar quantos traces já adicionamos para referência
    # (não obrigatório, só para entender a ordem)
    trace_idx = 1  # já temos 1 trace (dados reais)

    for grau, cor in zip(graus_lista, cores):
        poly = PolynomialFeatures(degree=grau)
        x_poly = poly.fit_transform(x_train)
        model = LinearRegression()
        model.fit(x_poly, y_train)

        x_all = np.concatenate([x_train, x_future])
        x_all_poly = poly.transform(x_all)
        y_all = model.predict(x_all_poly)
        dates_all = list(df_lancamentos_filtrado['AnoMes']) + future_dates

        visible_trace = True if grau == 3 else 'legendonly'

        # Parte histórica
        fig.add_trace(go.Scatter(
            x=dates_all[:len(x_train)],
            y=y_all[:len(x_train)],
            mode='lines',
            name=f'Regressão Grau {grau}',
            line=dict(color=cor, dash='solid'),
            visible=visible_trace
        ))
        trace_idx += 1

        # Parte futura (projeção)
        fig.add_trace(go.Scatter(
            x=dates_all[len(x_train):],
            y=y_all[len(x_train):],
            mode='lines',
            name=f'Projeção Futura (Grau {grau})',
            line=dict(color=cor, dash='dash'),
            visible=visible_trace
        ))
        trace_idx += 1

        # Monta equação
        coef = model.coef_
        intercept = model.intercept_
        equation = f"y = {intercept:.2f}"
        for i in range(1, grau + 1):
            equation += f" + {coef[i]:.2e}·x^{i}"

        # Annotation para a equação, só grau 3 visível inicialmente
        annotations.append(dict(
            text=f"<b>Regressão Grau {grau}:</b><br>{equation}",
            xref="paper", yref="paper",
            x=0.01, y=1.0 - grau * 0.12,
            xanchor='left', yanchor='top',
            showarrow=False,
            align="right",
            font=dict(size=12, color=cor),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            visible=(grau == 3)
        ))

    fig.update_layout(annotations=annotations)

    # Botões para mostrar/ocultar equações
    buttons = []
    for idx, grau in enumerate(graus_lista):
        # Criar uma cópia das anotações ajustando visibilidade para esse botão
        annotations_visibility = []
        for i, ann in enumerate(annotations):
            ann_copy = ann.copy()
            ann_copy['visible'] = (i == idx)
            annotations_visibility.append(ann_copy)
        buttons.append(dict(
            label=f"Mostrar Eq Grau {grau}",
            method="relayout",
            args=[{"annotations": annotations_visibility}]
        ))

    # Botão para ocultar todas
    annotations_off = []
    for ann in annotations:
        ann_copy = ann.copy()
        ann_copy['visible'] = False
        annotations_off.append(ann_copy)

    buttons.append(dict(
        label="Ocultar todas as equações",
        method="relayout",
        args=[{"annotations": annotations_off}]
    ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="down",
            showactive=True,
            x=1.15,
            y=0.8,
            buttons=buttons,
            xanchor='left',
            yanchor='top'
        )],

        annotations=annotations,  # mantém anotações originais iniciais

        title='Quantidade de Jogos Lançados (com Regressões e Projeções)',
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
    #--------------------------------------------------Avaliação de overfitting: Erro vs Grau do Polinômio-------------------------------------------------------#
    
    st.subheader("Avaliação de Overfitting: Erro por Grau do Polinômio")

    # Redivide os dados de lançamento em treino e validação (sem embaralhar por causa da série temporal)
    x = df_lancamentos_filtrado['AnoMes'].map(lambda d: d.toordinal() - base_ordinal).values.reshape(-1, 1)
    y = df_lancamentos_filtrado['Quantidade'].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

    graus = range(1, 7)
    train_errors = []
    val_errors = []

    for grau in graus:
        poly = PolynomialFeatures(degree=grau)
        x_train_poly = poly.fit_transform(x_train)
        x_val_poly = poly.transform(x_val)

        model = LinearRegression()
        model.fit(x_train_poly, y_train)

        y_train_pred = model.predict(x_train_poly)
        y_val_pred = model.predict(x_val_poly)

        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)

        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    fig_overfit = go.Figure()

    fig_overfit.add_trace(go.Scatter(
        x=list(graus),
        y=train_errors,
        mode='lines+markers',
        name='Erro de Treinamento',
        line=dict(color='blue')
    ))

    fig_overfit.add_trace(go.Scatter(
        x=list(graus),
        y=val_errors,
        mode='lines+markers',
        name='Erro de Validação',
        line=dict(color='red')
    ))

    fig_overfit.update_layout(
        title="Erro de Treinamento vs Validação por Grau do Polinômio",
        xaxis_title="Grau do Polinômio",
        yaxis_title="Erro RMSE",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    st.plotly_chart(fig_overfit, use_container_width=True)

    #-------------------------------------------Tabela--------------------------------------------------------------#

    # Preparar dados para a tabela de parâmetros e métricas
    from sklearn.metrics import mean_squared_error, r2_score

    # Reutiliza a divisão de treino/validação que já fez (x_train, x_val, y_train, y_val)
    results = []

    for grau, cor in zip(graus_lista, cores):
        poly = PolynomialFeatures(degree=grau)
        x_train_poly = poly.fit_transform(x_train)
        x_val_poly = poly.transform(x_val)

        model = LinearRegression()
        model.fit(x_train_poly, y_train)

        y_train_pred = model.predict(x_train_poly)
        y_val_pred = model.predict(x_val_poly)

        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        coef = model.coef_
        intercept = model.intercept_
        coef_str = ", ".join([f"{c:.2e}" for c in coef[1:]]) if grau > 0 else "N/A"

        results.append({
            "Grau": grau,
            "Intercepto": f"{intercept:.4f}",
            "Coeficientes": coef_str,
            "RMSE Treino": f"{train_rmse:.4f}",
            "RMSE Validação": f"{val_rmse:.4f}",
            "R² Treino": f"{train_r2:.4f}",
            "R² Validação": f"{val_r2:.4f}",
        })

    df_metrics = pd.DataFrame(results)

    st.subheader("Parâmetros e Métricas das Regressões Polinomiais")
    st.dataframe(df_metrics)

    st.markdown("""
    ### Legenda dos Parâmetros e Métricas

    - **Grau:** Grau do polinômio usado na regressão.
    - **Intercepto:** Valor de y quando todas as variáveis independentes são zero.
    - **Coeficientes:** Multiplicadores dos termos de grau 1, 2, ... na equação polinomial.
    - **RMSE Treino:** Raiz do erro quadrático médio no conjunto de treino (quanto menor, melhor).
    - **RMSE Validação:** Raiz do erro quadrático médio no conjunto de validação (quanto menor, melhor).
    - **R² Treino:** Coeficiente de determinação no treino, mede o quanto o modelo explica a variação dos dados (máximo 1, mais próximo de 1 é melhor).
    - **R² Validação:** Coeficiente de determinação na validação, indica a generalização do modelo (mais próximo de 1 é melhor).

    ---

    ### Qual o valor ideal para cada um?
                
    - **Grau do polinômio:** Nem sempre maior é melhor. Graus muito altos podem causar overfitting e piorar a generalização.
    - **RMSE:** Quanto menor, melhor. Indica erro médio do modelo em unidades originais (Nesse caso jogos/mês).
    - **R²:** Varia de 0 a 1 (pode ser negativo se o modelo é pior que uma média). Próximo de 1 indica bom ajuste.
    - **Diferença entre treino e validação:** Pequena diferença indica bom ajuste e boa capacidade de generalização; grandes diferenças indicam overfitting ou underfitting.


    """)
    
    #-------------------------------------------Gráfico do preço# --------------------------------------------------------------#

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
    preco_stats_filtrado['Faixa Inferior'] = (preco_stats_filtrado['Preço Médio'] - preco_stats_filtrado['Desvio Padrão']).clip(lower=0)

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
