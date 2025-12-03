import shap
import pandas as pd
import joblib
import os
import numpy as np

from api.digital_twin import load_or_train_model
from api.digital_twin_tools import CSV_PATH, SEG_CSV_OUT, MODEL_PATH, SCALER_PATH

# Evitar warnings
shap.initjs()

def load_model_and_data():
    """Carga modelo, scaler y dataset segmentado."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise ValueError("Modelo no entrenado. Entrena antes de usar SHAP.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(SEG_CSV_OUT)
    return model, scaler, df


def prepare_X(df):
    #Prepara las columnas exactamente como en el entrenamiento
    feature_cols = [
        "score_crediticio", "ingresos_mensuales", "uso_credito",
        "historial_impagos", "frecuencia_pago", "saldo_actual",
        "antiguedad_cliente", "cluster_kmeans"
    ]
    
    dummies = pd.get_dummies(df["comportamiento_app"], drop_first=True)
    X = pd.concat([df[feature_cols], dummies], axis=1)
    return X, X.columns.tolist()


def get_shap_explainer():
    #Devuelve el SHAP explainer y la data preparada.
    model, scaler, df = load_model_and_data()

    X, feature_cols = prepare_X(df)
    X_scaled = scaler.transform(X)

    explainer = shap.LinearExplainer(model, X_scaled)
    return explainer, X, X_scaled, df, feature_cols


def shap_for_client(client_id):
    #Devuelve SHAP values + explicación textual para un cliente.
    explainer, X, X_scaled, df, feature_names = get_shap_explainer()

    row = df[df["ID_cliente"] == client_id]
    if row.empty:
        return None, f"No se encontró el cliente {client_id}"

    idx = row.index[0]

    # Explicabilidad de SHAP
    shap_values = explainer.shap_values(X_scaled[idx:idx+1])

    # Convertir a DataFrame legible
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    }).sort_values("shap_value", ascending=False)

    return shap_df, explanation_summary(shap_df)


def explanation_summary(shap_df):
    #Convierte SHAP DF a explicación en lenguaje natural.
    top_positive = shap_df.head(3)
    top_negative = shap_df.tail(3)

    txt = "### Explicación SHAP del modelo de impago\n"

    txt += "\n**Factores que más aumentan la probabilidad de impago:**\n"
    for _, r in top_positive.iterrows():
        txt += f"- {r['feature']} (impacto +{r['shap_value']:.3f})\n"

    txt += "\n**Factores que reducen la probabilidad de impago:**\n"
    for _, r in top_negative.iterrows():
        txt += f"- {r['feature']} (impacto {r['shap_value']:.3f})\n"

    return txt

def shap_global():
    explainer, X, X_scaled, df, feature_names = get_shap_explainer()
    shap_values = explainer.shap_values(X_scaled)

    # Corrigiendo error de indices si es necesario
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return explainer, shap_values, X, df, feature_names
