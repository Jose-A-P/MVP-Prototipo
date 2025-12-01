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

# Carpeta base de datos y modelos (un nivel arriba de api/)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CSV_PATH = os.path.join(BASE_DIR, "clientes_sinteticos.csv")
SEG_CSV_OUT = os.path.join(BASE_DIR, "clientes_segmentados.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_logistico_impago.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_modelo.pkl")

RANDOM_STATE = 42

# === DEBUG: listar contenido de BASE_DIR ===
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

# Genera datos sintéticos
def generar_datos_sinteticos(n=5000, save_csv=True):
    np.random.seed(RANDOM_STATE)
    segmentos = np.random.choice(
        ["bajo_riesgo", "medio_riesgo", "alto_riesgo"],
        size=n,
        p=[0.40, 0.35, 0.25]
    )
    # ... [resto de la generación igual] ...
    df = pd.DataFrame({
        "ID_cliente": [f"C{i:05d}" for i in range(n)],
        "segmento_oculto": segmentos,
        # demás columnas...
    })
    if save_csv:
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Dataset sintético guardado en {CSV_PATH}")
    return df

def train_model(df=None, k_optimo=3, save_csv=True):
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

def load_or_train_model(df=None):
    features = [
        "score_crediticio","ingresos_mensuales","uso_credito",
        "historial_impagos","frecuencia_pago","saldo_actual",
        "antiguedad_cliente","cluster_kmeans"
    ]
    X = pd.concat([df[features], pd.get_dummies(df["comportamiento_app"], drop_first=True)], axis=1)
    y = df["impago"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Modelo entrenado y guardado en {MODEL_PATH} y {SCALER_PATH}")
    return model, scaler, X.columns

def predict_prob(model, scaler, df, feature_cols):
    X = pd.concat([df[feature_cols], pd.get_dummies(df["comportamiento_app"], drop_first=True)], axis=1)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X.reindex(columns=feature_cols + [c for c in X.columns if c not in feature_cols], fill_value=0)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]

# Escenarios (igual que antes)
def scenario_income_drop(df, pct_drop=0.2): ...
def scenario_interest_rate_increase(df, extra_rate=0.03): ...
def scenario_worse_payment_behavior(df, extra_missed=0.5): ...
def scenario_combined(df, income_drop=0.2, rate_inc=0.03, extra_missed=0.5): ...

# Resumen
def summarize_impact(df_base, df_scenario, probs_base, probs_scenario, top_n=10):
    df_res = df_base[["ID_cliente","cluster_kmeans"]].copy()
    df_res["prob_base"] = probs_base
    df_res["prob_scenario"] = probs_scenario
    df_res["delta_prob"] = df_res["prob_scenario"] - df_res["prob_base"]
    overall = {
        "mean_prob_base": df_res["prob_base"].mean(),
        "mean_prob_scenario": df_res["prob_scenario"].mean(),
        "mean_delta": df_res["delta_prob"].mean(),
        "pct_increase_mean": (df_res["prob_scenario"].mean() - df_res["prob_base"].mean()) / max(df_res["prob_base"].mean(), 1e-9)
    }
    by_cluster = df_res.groupby("cluster_kmeans").agg({
        "prob_base":"mean","prob_scenario":"mean","delta_prob":"mean","ID_cliente":"count"
    }).rename(columns={"ID_cliente":"n_clients"})
    top_risers = df_res.sort_values("delta_prob", ascending=False).head(top_n)
    return df_res, overall, by_cluster, top_risers

# Demo
def run_demo(do_export=False):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"[INFO] Cargado {CSV_PATH}")
    else:
        print("[WARN] No se encontró dataset. Generando dataset sintético...")
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
    df_res, overall, by_cluster, top_risers = summarize_impact(df, df_combined, probs_base, probs_comb)
    if do_export:
        df_out = df.copy()
        df_out["prob_base"] = probs_base
        df_out["prob_combined"] = probs_comb
        df_out.to_csv(os.path.join(BASE_DIR,"clientes_simulados_combined.csv"), index=False)
        df_res.to_csv(os.path.join(BASE_DIR,"impacto_combined_por_cliente.csv"), index=False)
        print("[INFO] Exportados clientes_simulados_combined.csv e impacto_combined_por_cliente.csv")
    return {"df_base":df,"df_combined":df_combined,"probs_base":probs_base,"probs_combined":probs_comb,"summary":overall,"by_cluster":by_cluster,"top_risers":top_risers}
