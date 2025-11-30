from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Cargar modelo y scaler
model = joblib.load("modelo_logistico_impago.pkl")
scaler = joblib.load("scaler_modelo.pkl")

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

@app.post("/predict")
def predict(client: dict):
    df = pd.DataFrame([client])
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    return {"prob_impago": float(prob)}

@app.post("/scenario")
def scenario(params: dict):
    # Aquí puedes reutilizar las funciones de DigitalTwing.ipynb
    # para aplicar income_drop, rate_inc, etc.
    return {"message": "Simulación de escenario en construcción"}
