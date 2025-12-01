from fastapi import FastAPI, HTTPException
import os
import pandas as pd
import joblib

from api.digital_twin import (
    generar_datos_sinteticos,
    train_model,
    load_or_train_model,
    predict_prob,
    scenario_income_drop,
    scenario_interest_rate_increase,
    scenario_worse_payment_behavior,
    scenario_combined,
    summarize_impact
)
from api.llm_gen import generate_summary

app = FastAPI(title="Digital Twin API", version="1.0")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(BASE_DIR, "clientes_sinteticos.csv")
SEG_CSV_OUT = os.path.join(BASE_DIR, "clientes_segmentados.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_logistico_impago.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_modelo.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-data")
def generate_data(n: int = 5000):
    df = generar_datos_sinteticos(n=n, save_csv=True)
    return {"rows": len(df), "path": CSV_PATH}

@app.post("/train-model")
def train_model_endpoint():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=400, detail="No existe dataset base")
    df = pd.read_csv(CSV_PATH)
    df = train_model(df, k_optimo=6, save_csv=True)
    model, scaler, feature_cols = load_or_train_model(df)
    return {"message": "Modelo entrenado", "features": list(feature_cols)}

@app.post("/simulate/{scenario}")
def simulate(scenario: str):
    # cargar dataset
    if os.path.exists(SEG_CSV_OUT):
        df = pd.read_csv(SEG_CSV_OUT)
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if "cluster_kmeans" not in df.columns:
            df = train_model(df, k_optimo=6, save_csv=True)
    else:
        raise HTTPException(status_code=400, detail="No existe dataset base")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="No existe modelo entrenado")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    feature_cols = [
        "score_crediticio","ingresos_mensuales","uso_credito",
        "historial_impagos","frecuencia_pago","saldo_actual",
        "antiguedad_cliente","cluster_kmeans"
    ]

    probs_base = predict_prob(model, scaler, df, feature_cols)

    # aplicar escenario
    if scenario == "income":
        df_scenario = scenario_income_drop(df, pct_drop=0.2)
    elif scenario == "rate":
        df_scenario = scenario_interest_rate_increase(df, extra_rate=0.1)
    elif scenario == "behavior":
        df_scenario = scenario_worse_payment_behavior(df, extra_missed=0.7)
    elif scenario == "combined":
        df_scenario = scenario_combined(df, income_drop=0.2, rate_inc=0.03, extra_missed=0.7)
    else:
        raise HTTPException(status_code=400, detail="Escenario inválido")

    probs_scenario = predict_prob(model, scaler, df_scenario, feature_cols)
    df_res, overall, by_cluster, top_risers = summarize_impact(df, df_scenario, probs_base, probs_scenario)

    # exportar CSVs para Streamlit
    df_out = df.copy()
    df_out["prob_base"] = probs_base
    df_out["prob_scenario"] = probs_scenario
    df_out.to_csv(os.path.join(BASE_DIR, "clientes_simulados_combined.csv"), index=False)
    df_res.to_csv(os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv"), index=False)

    return {
        "overall": overall,
        "by_cluster": by_cluster.reset_index().to_dict(orient="records"),
        "top_risers": top_risers.to_dict(orient="records"),
        "exported_files": ["clientes_simulados_combined.csv", "impacto_combined_por_cliente.csv"]
    }

@app.post("/report/summary")
def report_summary():
    try:
        df_res = pd.read_csv(os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv"))
        summary = generate_summary(df_res)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
