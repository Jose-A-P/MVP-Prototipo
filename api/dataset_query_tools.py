import pandas as pd
import numpy as np
import re

"""
Este módulo contiene funciones que permiten interpretar consultas
del usuario sobre el dataset procesado. Incluye:

1. Utilidades para extraer parámetros desde texto del usuario.
2. Funciones de análisis estadístico sobre el dataset.
3. Consultas relacionadas con riesgo de crédito.
4. Un router que interpreta el lenguaje natural y determina
   el tipo de consulta tabular a ejecutar.

Todas las funciones están diseñadas para ser utilizadas
por el router principal del chatbot.
"""


# ============================================================
# Funciones auxiliares para extracción de parámetros del texto
# ============================================================

def extract_top_n(prompt):
    """
    Extrae un número entero desde el texto del usuario.
    Se usa para consultas como "top 5" o "los 10 más afectados".
    """
    match = re.search(r'\b(\d+)\b', prompt)
    if match:
        return int(match.group(1))
    return None


def extract_cluster(prompt):
    """
    Detecta consultas que mencionan un cluster específico.
    Ejemplo: "cluster 2" o "clientes del cluster 1".
    """
    match = re.search(r'cluster\s*(\d+)', prompt)
    if match:
        return int(match.group(1))
    return None


def extract_threshold(prompt, df):
    """
    Detecta consultas que contienen un filtro numérico como:
    mayor a 0.4
    mayores a 2000
    > 0.3

    Determina automáticamente la columna a filtrar
    según el contexto semántico del prompt.

    Devuelve un par (columna, valor) o (None, None).
    """

    # Detecta el valor numérico utilizado como umbral
    match = re.search(r'(mayor a|mayores a|> ?)(\d+\.?\d*)', prompt)
    if not match:
        return None, None

    threshold = float(match.group(2))
    p = prompt.lower()

    # Selección de la columna basada en palabras clave
    if "riesgo" in p or "probabilidad" in p:
        return "prob_scenario", threshold
    if "ingreso" in p:
        return "ingresos_mensuales", threshold
    if "saldo" in p:
        return "saldo_actual", threshold
    if "uso" in p:
        return "uso_credito", threshold

    return None, None


# ============================================================
# Funciones estadísticas generales del dataset
# ============================================================

def get_mean(df, column):
    """
    Retorna la media de una columna del dataset.
    """
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df[column].mean()


def get_max(df, column):
    """
    Retorna la fila del cliente con el máximo valor en la columna dada.
    """
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df.loc[df[column].idxmax()]


def get_min(df, column):
    """
    Retorna la fila del cliente con el mínimo valor en la columna dada.
    """
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df.loc[df[column].idxmin()]


def describe_column(df, column):
    """
    Genera estadísticas descriptivas de una columna.
    """
    if column not in df.columns:
        return f"La columna '{column}' no existe en el dataset."
    return df[column].describe()


# ============================================================
# Funciones de análisis de riesgo
# ============================================================

def get_portfolio_risk_summary(df):
    """
    Produce un resumen de variables clave de riesgo:
    probabilidad base, probabilidad del escenario,
    deterioro promedio y porcentaje de clientes en deterioro.
    """
    return {
        "probabilidad_media_base": df["prob_base"].mean(),
        "probabilidad_media_escenario": df["prob_scenario"].mean(),
        "deterioro_promedio": df["delta_prob"].mean(),
        "clientes_deteriorados_%": (df["delta_prob"] > 0).mean()
    }


def top_n_by_delta(df, n=10):
    """
    Devuelve los n clientes más afectados por el escenario,
    ordenados por delta_prob descendente.
    """
    if "delta_prob" not in df.columns:
        df["delta_prob"] = df["prob_scenario"] - df["prob_base"]
    return df.sort_values("delta_prob", ascending=False).head(n)


def worst_clients(df, n=10):
    """
    Devuelve los n clientes con mayor prob_scenario.
    """
    return df.sort_values("prob_scenario", ascending=False).head(n)


def cluster_stats(df):
    """
    Genera estadísticas agregadas por cluster.
    Incluye medias de prob_base, prob_scenario, delta_prob y número de clientes.
    """
    return df.groupby("cluster_kmeans").agg({
        "prob_base": "mean",
        "prob_scenario": "mean",
        "delta_prob": "mean",
        "ID_cliente": "count"
    }).rename(columns={"ID_cliente": "n_clientes"})


def filter_by_risk(df, threshold=0.3):
    """
    Filtra clientes con prob_scenario >= threshold.
    """
    return df[df["prob_scenario"] >= threshold]


def get_client_info(df, client_id):
    """
    Devuelve un diccionario con la información completa de un cliente.
    """
    row = df[df["ID_cliente"] == client_id]
    if row.empty:
        return f"No se encontró el cliente {client_id}"
    return row.iloc[0].to_dict()


def apply_filters(df, cluster=None, column=None, threshold=None):
    """
    Aplica filtros combinados por cluster y columna.
    Se utiliza para consultas tabulares complejas del chatbot.
    """
    df2 = df.copy()

    if cluster is not None:
        df2 = df2[df2["cluster_kmeans"] == cluster]

    if column is not None and threshold is not None:
        df2 = df2[df2[column] >= threshold]

    return df2


# ============================================================
# Inteligencia para interpretar consultas tabulares
# ============================================================

def parse_dataset_query(prompt, df):
    """
    Interpreta lenguaje natural y determina si el usuario
    está solicitando:

    - Media
    - Máximo
    - Mínimo
    - Estadísticas descriptivas
    - Top deterioro
    - Riesgo por cluster
    - Filtrado por riesgo
    - Información de un cliente

    Retorna un diccionario con la intención detectada.
    """

    p = prompt.lower()

    # Detección de columnas mencionadas explícitamente
    columns_detected = []
    for col in df.columns:
        if col.lower() in p:
            columns_detected.append(col)

    # Detección de operaciones matemáticas y estadísticas
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

    # Consultas sobre riesgo por cluster
    if "cluster" in p and "riesgo" in p:
        return {"intent": "cluster_risk", "columns": []}

    if "clientes en riesgo" in p:
        return {"intent": "risk_filter", "columns": []}

    # Información de cliente por ID
    match = re.search(r"C\d{5}", p)
    if match:
        return {"intent": "client_info", "client_id": match.group(0)}

    # Si no se detecta ninguna intención
    return {"intent": None}
