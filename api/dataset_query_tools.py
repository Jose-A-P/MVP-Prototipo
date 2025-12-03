import pandas as pd
import numpy as np
import re

# FUNCIONES AUXILIARES
def extract_top_n(prompt):
    match = re.search(r'\b(\d+)\b', prompt)
    if match:
        return int(match.group(1))
    return None


def extract_cluster(prompt):
    match = re.search(r'cluster\s*(\d+)', prompt)
    if match:
        return int(match.group(1))
    return None


def extract_threshold(prompt, df):
    """
    Detecta filtros de la forma:
    - mayor a 0.4
    - arriba de 2000
    - > 0.3
    y los aplica a la columna correcta según contexto de la pregunta.
    """
    # Detectar número
    match = re.search(r'(mayor a|mayores a|> ?)(\d+\.?\d*)', prompt)
    if not match:
        return None, None

    threshold = float(match.group(2))

    # Detectar columna implícita según palabras usadas
    p = prompt.lower()

    if "riesgo" in p or "probabilidad" in p:
        return "prob_scenario", threshold
    if "ingreso" in p:
        return "ingresos_mensuales", threshold
    if "saldo" in p:
        return "saldo_actual", threshold
    if "uso" in p:
        return "uso_credito", threshold

    return None, None

# FUNCIONES GENERALES

def get_mean(df, column):
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df[column].mean()

def get_max(df, column):
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df.loc[df[column].idxmax()]

def get_min(df, column):
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df.loc[df[column].idxmin()]

def describe_column(df, column):
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df[column].describe()

# CONSULTAS DE RIESGO

def get_portfolio_risk_summary(df):
    return {
        "probabilidad_media_base": df["prob_base"].mean(),
        "probabilidad_media_escenario": df["prob_scenario"].mean(),
        "deterioro_promedio": df["delta_prob"].mean(),
        "clientes_deteriorados_%": (df["delta_prob"] > 0).mean()
    }

def top_n_by_delta(df, n=10):
    if "delta_prob" not in df.columns:
        df["delta_prob"] = df["prob_scenario"] - df["prob_base"]
    return df.sort_values("delta_prob", ascending=False).head(n)


def worst_clients(df, n=10):
    return df.sort_values("prob_scenario", ascending=False).head(n)

def cluster_stats(df):
    return df.groupby("cluster_kmeans").agg({
        "prob_base": "mean",
        "prob_scenario": "mean",
        "delta_prob": "mean",
        "ID_cliente": "count"
    }).rename(columns={"ID_cliente": "n_clientes"})

def filter_by_risk(df, threshold=0.3):
    return df[df["prob_scenario"] >= threshold]

def get_client_info(df, client_id):
    row = df[df["ID_cliente"] == client_id]
    if row.empty:
        return f"No se encontró el cliente {client_id}"
    return row.iloc[0].to_dict()

def apply_filters(df, cluster=None, column=None, threshold=None):
    df2 = df.copy()

    if cluster is not None:
        df2 = df2[df2["cluster_kmeans"] == cluster]

    if column is not None and threshold is not None:
        df2 = df2[df2[column] >= threshold]

    return df2


# INTELIGENCIA PARA DETECTAR PREGUNTAS

def parse_dataset_query(prompt, df):
    p = prompt.lower()

    # 1. IDENTIFICAR COLUMNAS
    columns_detected = []
    for col in df.columns:
        if col.lower() in p:
            columns_detected.append(col)

    # 2. IDENTIFICAR TIPO DE OPERACIÓN
    if "promedio" in p or "media" in p or "mean" in p:
        return {"intent": "mean", "columns": columns_detected}

    if "max" in p or "mayor" in p:
        return {"intent": "max", "columns": columns_detected}

    if "min" in p or "menor" in p:
        return {"intent": "min", "columns": columns_detected}

    if "describe" in p or "resumen" in p or "estadistica" in p:
        return {"intent": "describe", "columns": columns_detected}

    if "top" in p or "más afectados" in p or "mas afectados" in p or "peores" in p:
        return {"intent": "top_delta"}

    if "cluster" in p and "riesgo" in p:
        return {"intent": "cluster_risk", "columns": []}

    if "clientes en riesgo" in p:
        return {"intent": "risk_filter", "columns": []}

    import re
    match = re.search(r"C\d{5}", p)
    if match:
        return {"intent": "client_info", "client_id": match.group(0)}

    return {"intent": None}

