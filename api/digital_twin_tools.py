import os
import numpy as np
import pandas as pd
import joblib

from api.digital_twin import (
    scenario_income_drop,
    scenario_interest_rate_increase,
    scenario_worse_payment_behavior,
    scenario_combined,
    predict_prob,
    load_or_train_model,
    train_model
)

MODULE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(MODULE_DIR, "..", "data"))

# Crear carpeta si no existe
os.makedirs(BASE_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "clientes_sinteticos.csv")
SEG_CSV_OUT = os.path.join(BASE_DIR, "clientes_segmentados.csv")
COMBINED_PATH = os.path.join(BASE_DIR, "clientes_simulados_combined.csv")
IMPACTO_PATH = os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_logistico_impago.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_modelo.pkl")

# Debug muy claro para ver rutas dentro del contenedor
print("\n=== DIGITAL TWIN PATH DEBUG ===")
print("MODULE_DIR:", MODULE_DIR)
print("BASE_DIR:", BASE_DIR)
print("CSV_PATH:", CSV_PATH)
print("FILES IN BASE_DIR:", os.listdir(BASE_DIR))
print("================================\n")


# Cargar modelo (o entrenarlo si no existe)

def load_model_and_scaler(df=None):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # columnas usadas durante el entrenamiento
        if df is not None:
            feature_cols = [
                "score_crediticio","ingresos_mensuales","uso_credito",
                "historial_impagos","frecuencia_pago","saldo_actual",
                "antiguedad_cliente","cluster_kmeans"
            ]
        return model, scaler, feature_cols
    else:
        print("[INFO] No existe modelo: entrenando...")
        return load_or_train_model(df)

# Ejecutar un escenario, calcular probabilidades y exportar CSV

def simulate_and_export(df, scenario_name, **kwargs):
    """
    Ejecuta un escenario, calcula prob_base y prob_scenario
    y exporta los csv listos para el RAG + Streamlit.
    """

    # cargar modelo
    model, scaler, feature_cols = load_model_and_scaler(df)

    # prob base
    probs_base = predict_prob(model, scaler, df, feature_cols)

    # aplicar escenario
    if scenario_name == "income":
        df_scenario = scenario_income_drop(df, **kwargs)
    elif scenario_name == "rate":
        df_scenario = scenario_interest_rate_increase(df, **kwargs)
    elif scenario_name == "behavior":
        df_scenario = scenario_worse_payment_behavior(df, **kwargs)
    elif scenario_name == "combined":
        df_scenario = scenario_combined(df, **kwargs)
    else:
        raise ValueError("Escenario inválido")

    # prob escenario
    probs_scenario = predict_prob(model, scaler, df_scenario, feature_cols)

    # exportar archivo combinado para Streamlit + RAG
    df_out = df_scenario.copy()
    df_out["prob_base"] = probs_base
    df_out["prob_scenario"] = probs_scenario

    df_out.to_csv(COMBINED_PATH, index=False)

    # export impacto
    df_res = pd.DataFrame({
        "ID_cliente": df["ID_cliente"],
        "cluster_kmeans": df["cluster_kmeans"],
        "prob_base": probs_base,
        "prob_scenario": probs_scenario,
        "delta_prob": probs_scenario - probs_base,
    })

    df_res.to_csv(IMPACTO_PATH, index=False)

    return df_res


# Función general para permitir al Chatbot ejecutar escenarios

def chatbot_execute_action(user_query):
    """Detecta intención del usuario y ejecuta el escenario adecuado."""
    
    if not os.path.exists(CSV_PATH):
        # autogenerar dataset faltante
        from api.digital_twin import generar_datos_sinteticos
        df = generar_datos_sinteticos(n=5000, save_csv=True)
    else:
        df = pd.read_csv(CSV_PATH)


    # clusterización si no existe
    if "cluster_kmeans" not in df.columns:
        df = train_model(df, k_optimo=6, save_csv=True)

    q = user_query.lower()

    # ===== Detectar intención ======
    if "escenario combinado" in q:
        result = simulate_and_export(df, "combined",
                                     income_drop=0.2,
                                     rate_inc=0.03,
                                     extra_missed=0.5)
        return "Escenario combinado ejecutado correctamente."

    if "escenario ingreso" in q or "bajar ingresos" in q:
        result = simulate_and_export(df, "income", pct_drop=0.2)
        return "Escenario de caída de ingresos ejecutado."

    if "escenario tasa" in q or "subir tasa" in q:
        result = simulate_and_export(df, "rate", extra_rate=0.03)
        return "Escenario de aumento de tasas ejecutado."

    if "escenario comportamiento" in q or "peor comportamiento" in q:
        result = simulate_and_export(df, "behavior", extra_missed=0.5)
        return "Escenario de comportamiento ejecutado."

    return None  # No detectó intención


# API para recargar datos desde Streamlit

def load_latest_data():
    #Carga el dataset combinado más reciente.
    if os.path.exists(COMBINED_PATH):
        df = pd.read_csv(COMBINED_PATH)
        if "prob_base" in df.columns and "prob_scenario" in df.columns:
                if "delta_prob" not in df.columns:
                    df["delta_prob"] = df["prob_scenario"] - df["prob_base"]

        return df
    return None
