import pandas as pd
import numpy as np

from api.digital_twin_tools import IMPACTO_PATH, COMBINED_PATH

def load_scenario_results():
    try:
        df = pd.read_csv(IMPACTO_PATH)
        return df
    except:
        return None


def compute_scenario_metrics():
    df = load_scenario_results()
    if df is None:
        return None

    metrics = {}

    metrics["mean_base"] = df["prob_base"].mean()
    metrics["mean_scenario"] = df["prob_scenario"].mean()
    metrics["mean_delta"] = df["delta_prob"].mean()

    metrics["max_delta"] = df["delta_prob"].max()
    metrics["min_delta"] = df["delta_prob"].min()

    # cluster más afectado
    cluster_stats = df.groupby("cluster_kmeans")["delta_prob"].mean().sort_values(ascending=False)
    metrics["cluster_worst"] = cluster_stats.index[0]
    metrics["cluster_worst_delta"] = cluster_stats.iloc[0]

    # top 5 clientes más afectados
    top5 = df.sort_values("delta_prob", ascending=False).head(5)[["ID_cliente", "delta_prob"]]
    metrics["top5"] = top5.to_dict(orient="records")

    # clasificación del impacto general (suave, moderado, severo)
    if metrics["mean_delta"] < 0.01:
        metrics["severity"] = "suave"
    elif metrics["mean_delta"] < 0.03:
        metrics["severity"] = "moderado"
    else:
        metrics["severity"] = "severo"

    return metrics
