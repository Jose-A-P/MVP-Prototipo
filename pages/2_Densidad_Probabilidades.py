import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from api.digital_twin_tools import load_latest_data

st.title("Análisis de Probabilidades – Base vs Escenario")

df = load_latest_data()

if df is None:
    st.warning("No hay datos combinados. Ejecuta un escenario desde el chatbot.")
    st.stop()

required_cols = ["prob_base", "prob_scenario"]

if not all(c in df.columns for c in required_cols):
    st.error("El dataset cargado no tiene probabilidades. Ejecuta un escenario primero.")
    st.stop()

# Crear delta_prob si no existe
if "delta_prob" not in df.columns:
    df["delta_prob"] = df["prob_scenario"] - df["prob_base"]

# KDE base vs escenario
st.subheader("📈 Densidad de probabilidad – Base vs Escenario")

base = df["prob_base"].dropna().values
scen = df["prob_scenario"].dropna().values

fig = ff.create_distplot(
    [base, scen],
    group_labels=["Base", "Escenario"],
    show_hist=False,
    show_rug=False
)

fig.update_layout(
    title="Distribución de probabilidad",
    xaxis_title="Probabilidad de impago",
    yaxis_title="Densidad"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# KDE por cluster
st.subheader("🎯 Densidad por Cluster")

clusters = sorted(df["cluster_kmeans"].unique())

for c in clusters:
    sub = df[df["cluster_kmeans"] == c]

    base = sub["prob_base"].values
    scen = sub["prob_scenario"].values

    st.markdown(f"### Cluster {c}")

    fig_c = ff.create_distplot(
        [base, scen],
        group_labels=["Base", "Escenario"],
        show_hist=False,
        show_rug=False
    )

    fig_c.update_layout(
        xaxis_title="Probabilidad de impago",
        yaxis_title="Densidad",
    )

    st.plotly_chart(fig_c, use_container_width=True)

st.divider()

# KDE delta_prob
st.subheader("Distribución de cambios en probabilidad (Delta)")

fig_delta = ff.create_distplot(
    [df["delta_prob"].dropna()],
    group_labels=["Δ probabilidad"],
    show_hist=False,
    show_rug=False
)

fig_delta.update_layout(
    title="Densidad de Δ (prob_scenario - prob_base)",
    xaxis_title="Cambio en probabilidad",
    yaxis_title="Densidad"
)

st.plotly_chart(fig_delta, use_container_width=True)

st.divider()

# Scatter interactivo
st.subheader("🔍 Scatter interactivo – Ingresos vs Probabilidad")

metric = st.selectbox(
    "Selecciona qué probabilidad graficar:",
    ["prob_scenario", "prob_base", "delta_prob"]
)

fig_scatter = px.scatter(
    df,
    x="ingresos_mensuales",
    y=metric,
    color="cluster_kmeans",
    hover_data=["ID_cliente"],
    title=f"Ingresos vs {metric}",
    opacity=0.65
)

fig_scatter.update_layout(
    xaxis_title="Ingresos mensuales",
    yaxis_title=metric
)

st.plotly_chart(fig_scatter, use_container_width=True)
