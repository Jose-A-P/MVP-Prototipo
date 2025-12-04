import pandas as pd
import numpy as np

from api.digital_twin_tools import IMPACTO_PATH, COMBINED_PATH


def load_scenario_results():
    # Carga el archivo de impacto generado por el escenario más reciente.
    try:
        df = pd.read_csv(IMPACTO_PATH)
        return df
    except:
        return None


def compute_scenario_metrics():
    # Calcula métricas agregadas del escenario ya procesado.
    # Las métricas permiten describir la severidad del shock y el comportamiento del portafolio.

    df = load_scenario_results()
    if df is None:
        return None

    metrics = {}

    # Cálculo de probabilidades promedio antes y después del escenario.
    metrics["mean_base"] = df["prob_base"].mean()
    metrics["mean_scenario"] = df["prob_scenario"].mean()
    metrics["mean_delta"] = df["delta_prob"].mean()

    # Valores extremos del deterioro.
    metrics["max_delta"] = df["delta_prob"].max()
    metrics["min_delta"] = df["delta_prob"].min()

    # Identificación del cluster con mayor deterioro promedio.
    cluster_stats = df.groupby("cluster_kmeans")["delta_prob"].mean().sort_values(ascending=False)
    metrics["cluster_worst"] = cluster_stats.index[0]
    metrics["cluster_worst_delta"] = cluster_stats.iloc[0]

    # Selección de los cinco clientes más deteriorados según delta_prob.
    top5 = df.sort_values("delta_prob", ascending=False).head(5)[["ID_cliente", "delta_prob"]]
    metrics["top5"] = top5.to_dict(orient="records")

    # Clasificación de severidad basada en el deterioro promedio.
    if metrics["mean_delta"] < 0.01:
        metrics["severity"] = "suave"
    elif metrics["mean_delta"] < 0.03:
        metrics["severity"] = "moderado"
    else:
        metrics["severity"] = "severo"

    # Las métricas resultantes sirven como insumo para RAG, reportes ejecutivos y explicaciones del bot.
    return metrics
