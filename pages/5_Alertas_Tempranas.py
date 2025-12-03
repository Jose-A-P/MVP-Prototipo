import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from api.digital_twin_tools import load_latest_data

st.set_page_config(layout="wide")
st.title("🚨 Motor de Alertas Tempranas – Escenario Simulado")

# CARGA DE DATOS
df = load_latest_data()

if df is None or "prob_scenario" not in df.columns:
    st.warning("No hay datos combinados. Ejecuta un escenario desde el chatbot.")
    st.stop()

# CREAR prob_base SI NO EXISTE
if "prob_base" in df.columns and "prob_scenario" in df.columns:
    if "delta_prob" not in df.columns:
        df["delta_prob"] = df["prob_scenario"] - df["prob_base"]
else:
    st.error("El dataset cargado no tiene columnas prob_base / prob_scenario.")
    st.stop()

# VALIDA CLUSTERS E INDICA SI NO EXISTEN
if "cluster_kmeans" not in df.columns:
    st.error("El archivo cargado no incluye la segmentación (cluster_kmeans). Ejecuta el pipeline completo.")
    st.stop()

# CREACIÓN DE UMRALES
st.sidebar.header("Configuración del Motor EWS")

umbral_riesgo_final = st.sidebar.slider(
    "Umbral de prob_scenario para alerta crítica", 
    0.1, 1.0, 0.4, 0.05
)

umbral_delta = st.sidebar.slider(
    "Umbral de incremento de riesgo (delta_prob)", 
    0.0, 1.0, 0.20, 0.01
)

# Clasificación de severidad
def clasificar_alerta(row):
    if row["prob_scenario"] >= umbral_riesgo_final and row["delta_prob"] >= umbral_delta:
        return "CRÍTICA"
    elif row["delta_prob"] >= umbral_delta:
        return "ALTA"
    elif row["prob_scenario"] >= umbral_riesgo_final:
        return "MEDIA"
    return "NORMAL"

df["alerta"] = df.apply(clasificar_alerta, axis=1)

# MÉTRICAS GLOBALES
st.subheader("Resumen del Portafolio")

col1, col2, col3 = st.columns(3)

col1.metric("Probabilidad media (Base)", f"{df['prob_base'].mean():.3f}")
col2.metric("Probabilidad media (Escenario)", f"{df['prob_scenario'].mean():.3f}")
col3.metric("Incremento promedio", f"{df['delta_prob'].mean():.3f}")

col1, col2, col3 = st.columns(3)

col1.metric("Clientes con deterioro", f"{(df['delta_prob'] > umbral_delta).mean()*100:.1f}%")
col2.metric("Alertas críticas", f"{(df['alerta'] == 'CRÍTICA').sum()}")
col3.metric("Alertas totales", f"{(df['alerta'] != 'NORMAL').sum()}")


st.divider()

# ALERTAS POR CLUSTER
st.subheader("Alertas por Cluster")

cluster_stats = df.groupby("cluster_kmeans").agg({
    "delta_prob": "mean",
    "prob_scenario": "mean",
    "ID_cliente": "count",
    "alerta": lambda x: (x != "NORMAL").sum()
}).rename(columns={"ID_cliente": "n_clientes", "alerta": "alertas"})

st.dataframe(cluster_stats.style.background_gradient(cmap="coolwarm"), use_container_width=True)

fig_cluster = px.bar(
    cluster_stats,
    x=cluster_stats.index,
    y="delta_prob",
    color="alertas",
    title="Impacto Promedio por Cluster",
    labels={"cluster_kmeans": "Cluster", "delta_prob": "Incremento promedio"}
)

st.plotly_chart(fig_cluster, use_container_width=True)

st.divider()


# TABLA PRINCIPAL DE ALERTAS
st.subheader("Clientes con Alertas")

alertas_df = df[df["alerta"] != "NORMAL"].sort_values("delta_prob", ascending=False)

st.dataframe(
    alertas_df[
        ["ID_cliente", "cluster_kmeans", "prob_base", "prob_scenario", "delta_prob", "alerta"]
    ],
    use_container_width=True
)


# SUGERENCIAS AUTOMÁTICAS
st.subheader("Recomendaciones del Sistema")

sugerencias = []

if (df["alerta"] == "CRÍTICA").sum() > 0:
    sugerencias.append("- Priorizar contacto inmediato con clientes críticos.")
if cluster_stats["alertas"].max() > 5:
    cluster_peor = cluster_stats["alertas"].idxmax()
    sugerencias.append(f"- Revisar el cluster {cluster_peor}, elevado deterioro.")
if df["delta_prob"].mean() > 0.05:
    sugerencias.append("- Considerar medidas macro: límites, tasas, scoring más conservador.")
if df["prob_scenario"].mean() > df["prob_base"].mean() * 1.3:
    sugerencias.append("- La morosidad proyectada aumentó más del 30%: activar comité de riesgo.")

if not sugerencias:
    sugerencias.append("No se detectaron riesgos relevantes en el escenario actual.")

for s in sugerencias:
    st.markdown(f"- {s}")
