import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Este módulo implementa la construcción completa del Digital Twin financiero:

1. Generación de datos sintéticos para simulación.
2. Entrenamiento de un modelo de segmentación mediante KMeans.
3. Entrenamiento de un modelo de predicción de probabilidad de impago
   basado en regresión logística.
4. Funciones para ejecución de escenarios de estrés (income drop,
   aumento de tasas y deterioro del comportamiento de pago).
5. Implementación de la predicción de probabilidades usando el modelo entrenado.
6. Funciones de resumen que cuantifican el impacto del escenario en el portafolio.

Este módulo contiene la lógica central del Digital Twin, y sirve como base
para otros componentes como el chatbot, el motor de RAG y el reporte ejecutivo.
"""


# ============================================================
# Rutas de archivos de datos y modelos
# ============================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CSV_PATH = os.path.join(BASE_DIR, "clientes_sinteticos.csv")
SEG_CSV_OUT = os.path.join(BASE_DIR, "clientes_segmentados.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_logistico_impago.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_modelo.pkl")

RANDOM_STATE = 42


# ============================================================
# Debug opcional para inspeccionar el contenedor o entorno
# ============================================================

try:
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    if os.path.exists(BASE_DIR):
        print("[DEBUG] Archivos en BASE_DIR:")
        for f in os.listdir(BASE_DIR):
            print("   -", f)
    else:
        print("[DEBUG] La carpeta BASE_DIR no existe en el contenedor")
except Exception as e:
    print("[DEBUG] Error al listar BASE_DIR:", e)


# ============================================================
# Generación de datos sintéticos
# ============================================================

def generar_datos_sinteticos(n=5000, save_csv=True):
    """
    Genera un dataset sintético que simula clientes de crédito con diferentes niveles
    de riesgo, ingresos, uso de crédito y comportamiento histórico.
    """

    np.random.seed(RANDOM_STATE)

    segmentos = np.random.choice(
        ["bajo_riesgo", "medio_riesgo", "alto_riesgo"],
        size=n,
        p=[0.40, 0.35, 0.25]
    )

    df = pd.DataFrame({
        "ID_cliente": [f"C{i:05d}" for i in range(n)],
        "segmento_oculto": segmentos
        # En la implementación real aquí se incluyen todas las variables de crédito.
    })

    if save_csv:
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Dataset sintético guardado en {CSV_PATH}")

    return df


# ============================================================
# Segmentación con KMeans
# ============================================================

def train_model(df=None, k_optimo=3, save_csv=True):
    """
    Entrena un modelo KMeans utilizando las variables financieras principales.
    Añade una columna cluster_kmeans al DataFrame.
    """

    features = [
        "score_crediticio","ingresos_mensuales","uso_credito",
        "historial_impagos","frecuencia_pago","saldo_actual","antiguedad_cliente"
    ]

    df_encoded = pd.get_dummies(df[["comportamiento_app"]], drop_first=True)
    X = pd.concat([df[features], df_encoded], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42)
    df["cluster_kmeans"] = kmeans_final.fit_predict(X_scaled)

    if save_csv:
        df.to_csv(SEG_CSV_OUT, index=False)
        print(f"[INFO] Archivo generado: {SEG_CSV_OUT}")

    return df


# ============================================================
# Entrenar o cargar modelo de regresión logística para impago
# ============================================================

def load_or_train_model(df=None):
    """
    Entrena un modelo de regresión logística para predecir la probabilidad de impago.
    Este modelo es la base del Digital Twin para calcular prob_scenario y prob_base.
    """

    features = [
        "score_crediticio","ingresos_mensuales","uso_credito",
        "historial_impagos","frecuencia_pago","saldo_actual",
        "antiguedad_cliente","cluster_kmeans"
    ]

    X = pd.concat([df[features], pd.get_dummies(df["comportamiento_app"], drop_first=True)], axis=1)
    y = df["impago"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[INFO] Modelo entrenado y guardado en {MODEL_PATH} y {SCALER_PATH}")

    return model, scaler, X.columns


# ============================================================
# Predicción de probabilidad de impago
# ============================================================

def predict_prob(model, scaler, df, feature_cols):
    """
    Calcula probabilidad de impago para cada cliente en el DataFrame.

    """

    X = pd.concat(
        [df[feature_cols], pd.get_dummies(df["comportamiento_app"], drop_first=True)],
        axis=1
    )

    full_cols = scaler.feature_names_in_

    for col in full_cols:
        if col not in X.columns:
            X[col] = 0

    X = X.reindex(columns=full_cols)

    X_scaled = scaler.transform(X)

    return model.predict_proba(X_scaled)[:, 1]


# ============================================================
# Escenarios del Digital Twin
# ============================================================

def scenario_income_drop(df, pct_drop=0.2):
    """
    Reduce los ingresos mensuales en un porcentaje dado.
    Simula pérdida de capacidad de pago.
    """
    df2 = df.copy()
    df2["ingresos_mensuales"] *= (1 - pct_drop)
    return df2


def scenario_interest_rate_increase(df, extra_rate=0.03):
    """
    Simula un aumento en las tasas de interés.
    Eleva el saldo actual y el uso de crédito.
    """
    df2 = df.copy()
    df2["saldo_actual"] *= (1 + extra_rate * 2)
    df2["uso_credito"] = (df2["uso_credito"] + extra_rate * 3).clip(0, 1)
    return df2


def scenario_worse_payment_behavior(df, extra_missed=0.5):
    """
    Incrementa impagos históricos y reduce frecuencia de pago,
    representando deterioro del comportamiento.
    """
    df2 = df.copy()
    df2["historial_impagos"] = (
        df2["historial_impagos"] +
        np.random.poisson(lam=extra_missed, size=len(df2))
    ).astype(int)

    df2["frecuencia_pago"] = (df2["frecuencia_pago"] - extra_missed).clip(0)

    return df2


def scenario_combined(df, income_drop=0.2, rate_inc=0.03, extra_missed=0.5):
    """
    Aplica los tres deterioros al mismo tiempo.
    Representa un escenario macroeconómico severo.
    """
    df2 = scenario_income_drop(df, income_drop)
    df2 = scenario_interest_rate_increase(df2, rate_inc)
    df2 = scenario_worse_payment_behavior(df2, extra_missed)
    return df2


# ============================================================
# Resumen del impacto de un escenario
# ============================================================

def summarize_impact(df_base, df_scenario, probs_base, probs_scenario, top_n=10):
    """
    Calcula estadísticas principales del impacto del escenario en el portafolio.
    """

    df_res = df_base[["ID_cliente","cluster_kmeans"]].copy()
    df_res["prob_base"] = probs_base
    df_res["prob_scenario"] = probs_scenario
    df_res["delta_prob"] = df_res["prob_scenario"] - df_res["prob_base"]

    overall = {
        "mean_prob_base": df_res["prob_base"].mean(),
        "mean_prob_scenario": df_res["prob_scenario"].mean(),
        "mean_delta": df_res["delta_prob"].mean(),
        "pct_increase_mean": (
            df_res["prob_scenario"].mean() - df_res["prob_base"].mean()
        ) / max(df_res["prob_base"].mean(), 1e-9)
    }

    by_cluster = df_res.groupby("cluster_kmeans").agg({
        "prob_base": "mean",
        "prob_scenario": "mean",
        "delta_prob": "mean",
        "ID_cliente": "count"
    }).rename(columns={"ID_cliente":"n_clients"})

    top_risers = df_res.sort_values("delta_prob", ascending=False).head(top_n)

    return df_res, overall, by_cluster, top_risers


# ============================================================
# Ejecución de una demostración completa
# ============================================================

def run_demo(do_export=False):

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"[INFO] Cargado {CSV_PATH}")
    else:
        print("[WARN] No se encontró dataset. Generando dataset sintético.")
        df = generar_datos_sinteticos(n=5000, save_csv=True)

    if "cluster_kmeans" not in df.columns:
        df = train_model(df, k_optimo=6, save_csv=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        feature_cols = [
            "score_crediticio","ingresos_mensuales","uso_credito",
            "historial_impagos","frecuencia_pago","saldo_actual",
            "antiguedad_cliente","cluster_kmeans"
        ]
    else:
        model, scaler, feature_cols = load_or_train_model(df)

    probs_base = predict_prob(model, scaler, df, feature_cols)

    df_combined = scenario_combined(df, income_drop=0.2, rate_inc=0.03, extra_missed=0.7)
    probs_comb = predict_prob(model, scaler, df_combined, feature_cols)

    df_res, overall, by_cluster, top_risers = summarize_impact(
        df, df_combined, probs_base, probs_comb
    )

    if do_export:
        df_out = df.copy()
        df_out["prob_base"] = probs_base
        df_out["prob_combined"] = probs_comb
        df_out.to_csv(os.path.join(BASE_DIR, "clientes_simulados_combined.csv"), index=False)
        df_res.to_csv(os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv"), index=False)
        print("[INFO] Exportados clientes_simulados_combined.csv e impacto_combined_por_cliente.csv")

    return {
        "df_base": df,
        "df_combined": df_combined,
        "probs_base": probs_base,
        "probs_combined": probs_comb,
        "summary": overall,
        "by_cluster": by_cluster,
        "top_risers": top_risers
    }
