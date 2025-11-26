import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# CONFIG
CSV_PATH = "clientes_simulados_combined.csv"
MODEL_PATH = "modelo_logistico_impago.pkl"
SCALER_PATH = "scaler_modelo.pkl"

# CARGAR DATA & MODELO
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

df = load_data()
model, scaler = load_model()

# BASE FEATURES DEL MODELO
FEATURES = [
    "score_crediticio",
    "ingresos_mensuales",
    "uso_credito",
    "historial_impagos",
    "frecuencia_pago",
    "saldo_actual",
    "antiguedad_cliente",
    "cluster_kmeans"
]

# FUNCIONES PARA GENERAR LOS ESCENARIOS
def apply_income_drop(df, pct):
    d = df.copy()
    d["ingresos_mensuales"] *= (1 - pct)
    return d

def apply_interest_increase(df, rate):
    d = df.copy()
    d["saldo_actual"] *= (1 + rate * 2)
    d["uso_credito"] = (d["uso_credito"] + rate * 3).clip(0,1)
    return d

def apply_worse_behavior(df, extra):
    d = df.copy()
    d["historial_impagos"] = (d["historial_impagos"] + np.random.poisson(extra, len(df))).astype(int)
    d["frecuencia_pago"] = (d["frecuencia_pago"] - extra).clip(0)
    return d

def apply_combined(df, income_drop, rate_inc, extra_missed):
    d = apply_income_drop(df, income_drop)
    d = apply_interest_increase(d, rate_inc)
    d = apply_worse_behavior(d, extra_missed)
    return d

# PREDECIR LAS PROBABILIDADES CON LOS CAMBIOS
def predict(df):
    X = pd.concat(
        [df[FEATURES], pd.get_dummies(df["comportamiento_app"], drop_first=True)],
        axis=1
    )

    # Asegurar columnas faltantes (en caso de categorías)
    for col in ["medio", "bajo"]:
        if col not in X.columns:
            X[col] = 0

    X = X.reindex(columns=X.columns, fill_value=0)

    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]

# STREAMLIT UI
st.set_page_config(page_title="Digital Twin Financiero", layout="wide")
st.title("Digital Twin Financiero – Simulación de Escenarios")
st.markdown("Dashboard que permite evaluar **cómo cambian las probabilidades de impago** bajo distintos escenarios económicos.")

st.sidebar.header("Configuración de Escenario")

income_drop = st.sidebar.slider(
    "Caída de ingresos (%)",
    0.0, 0.5, 0.2, step=0.05
)

rate_inc = st.sidebar.slider(
    "Aumento de tasas (%)",
    0.0, 0.1, 0.03, step=0.01
)

extra_missed = st.sidebar.slider(
    "Incremento en impagos esperados",
    0.0, 2.0, 0.7, step=0.1
)

# APLICAR EL ESCENARIO ELEGIDO
df_base = df.copy()
df_scenario = apply_combined(df, income_drop, rate_inc, extra_missed)

# Predicciones
df_base["prob_base"] = predict(df_base)
df_scenario["prob_scenario"] = predict(df_scenario)

df_res = df_base[["ID_cliente", "cluster_kmeans", "prob_base"]].copy()
df_res["prob_scenario"] = df_scenario["prob_scenario"]
df_res["delta"] = df_res["prob_scenario"] - df_res["prob_base"]

# Evaluando la probabilidad y su cambio
col1, col2, col3 = st.columns(3)

col1.metric(
    "Prob. Promedio Base",
    f"{df_res['prob_base'].mean():.3f}"
)

col2.metric(
    "Prob. Promedio Escenario",
    f"{df_res['prob_scenario'].mean():.3f}",
    delta=f"{df_res['prob_scenario'].mean() - df_res['prob_base'].mean():.3f}"
)

col3.metric(
    "Cambio Relativo (%)",
    f"{((df_res['prob_scenario'].mean() / df_res['prob_base'].mean()) - 1) * 100:.1f}%"
)

st.markdown("---")

# DISTRIBUCIONES
st.subheader("Distribución de Probabilidades de Impago")
fig, ax = plt.subplots(figsize=(10,5))
sns.kdeplot(df_res["prob_base"], label="Base", ax=ax)
sns.kdeplot(df_res["prob_scenario"], label="Escenario", ax=ax)
ax.set_title("Distribución Base vs Escenario")
ax.legend()
st.pyplot(fig)


# IMPACTO POR CLUSTER
st.subheader("Impacto por Cluster")

cluster_summary = df_res.groupby("cluster_kmeans")[["prob_base", "prob_scenario", "delta"]].mean()

st.dataframe(cluster_summary.style.background_gradient(cmap="coolwarm"))

fig2, ax2 = plt.subplots(figsize=(7,4))
sns.barplot(
    data=cluster_summary.reset_index(),
    x="cluster_kmeans",
    y="delta",
    palette="coolwarm",
    ax=ax2
)
ax2.set_title("Cambio medio por cluster")
st.pyplot(fig2)

# CLIENTE INDIVIDUAL
st.subheader("Analizar Cliente Individual")

cliente_id = st.selectbox(
    "Selecciona un cliente:",
    df_res["ID_cliente"].unique()
)

cliente = df_res[df_res["ID_cliente"] == cliente_id].iloc[0]

colA, colB, colC = st.columns(3)
colA.metric("Prob. Base", f"{cliente['prob_base']:.3f}")
colB.metric("Prob. Escenario", f"{cliente['prob_scenario']:.3f}")
colC.metric("Δ Cambio", f"{cliente['delta']:.3f}")

# CARACTERISTICAS DEL CLIENTE
st.write("Características del cliente:")
st.dataframe(df[df["ID_cliente"] == cliente_id])

st.markdown("---")


# TOP 10 MAS AFECTADOS
st.subheader("Top 10 clientes más afectados por el escenario")

top_risers = df_res.sort_values("delta", ascending=False).head(10)
st.dataframe(top_risers)

# DESCARGAR EL ESCENARIO ACTUAL
csv_export = df_res.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar resultados del escenario (CSV)",
    data=csv_export,
    file_name="digital_twin_resultados.csv",
    mime="text/csv"
)
