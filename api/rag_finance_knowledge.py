import pandas as pd
import numpy as np

from api.digital_twin_tools import load_latest_data
from api.scenario_explainer import compute_scenario_metrics


def build_variable_dictionary() -> dict:
    # Define un diccionario con la explicación conceptual de cada variable financiera.
    # Este diccionario sirve como fuente de conocimiento estático para el sistema RAG.
    return {
        "score_crediticio": "Puntaje de riesgo crediticio del cliente. A mayor score, menor riesgo.",
        "ingresos_mensuales": "Ingresos mensuales estimados del cliente.",
        "uso_credito": "Proporción de utilización del crédito respecto al límite disponible.",
        "historial_impagos": "Número de impagos históricos registrados.",
        "frecuencia_pago": "Frecuencia de pago del cliente, en una escala donde valores mayores implican pagos más constantes.",
        "saldo_actual": "Saldo actual adeudado por el cliente.",
        "antiguedad_cliente": "Antigüedad de la relación del cliente con la entidad, en meses.",
        "cluster_kmeans": "Cluster de segmentación obtenido a partir de características de riesgo y comportamiento.",
        "prob_base": "Probabilidad de impago estimada en el escenario base.",
        "prob_scenario": "Probabilidad de impago estimada bajo el escenario simulado.",
        "delta_prob": "Cambio en la probabilidad de impago (escenario menos base)."
    }


def build_portfolio_overview(df: pd.DataFrame) -> str:
    # Genera un resumen descriptivo del comportamiento agregado del portafolio.
    # La información presentada es utilizada como contexto para el motor RAG.
    n_clients = len(df)
    mean_base = df["prob_base"].mean() if "prob_base" in df.columns else np.nan
    mean_scen = df["prob_scenario"].mean() if "prob_scenario" in df.columns else np.nan
    mean_delta = df["delta_prob"].mean() if "delta_prob" in df.columns else np.nan

    high_risk = (df["prob_scenario"] >= 0.4).mean() if "prob_scenario" in df.columns else np.nan

    txt = f"""
    RESUMEN DEL PORTAFOLIO

    - Número total de clientes: {n_clients}
    - Probabilidad media de impago (escenario base): {mean_base:.4f}
    - Probabilidad media de impago (escenario simulado): {mean_scen:.4f}
    - Aumento promedio en probabilidad de impago (delta): {mean_delta:.4f}
    - Porcentaje de clientes en alto riesgo (prob_scenario >= 0.4): {high_risk*100:.2f}%

    Este resumen describe el comportamiento agregado del portafolio bajo el escenario actual.
    """
    return txt


def build_cluster_profiles(df: pd.DataFrame) -> str:
    # Genera descripciones separadas por cluster, resumiendo comportamientos promedio.
    # Es útil para análisis conceptuales que el RAG debe responder.
    if "cluster_kmeans" not in df.columns:
        return "No hay información de clusters en el dataset actual."

    lines = ["PERFILES DE CLUSTERS\n"]
    grouped = df.groupby("cluster_kmeans")

    for cl, g in grouped:
        size = len(g)
        mean_base = g["prob_base"].mean()
        mean_scen = g["prob_scenario"].mean()
        mean_delta = g["delta_prob"].mean()

        income = g["ingresos_mensuales"].mean()
        score = g["score_crediticio"].mean()
        saldo = g["saldo_actual"].mean()
        uso = g["uso_credito"].mean()

        lines.append(
            f"""
            Cluster {cl}:
            - Número de clientes: {size}
            - Probabilidad media base: {mean_base:.4f}
            - Probabilidad media escenario: {mean_scen:.4f}
            - Delta promedio: {mean_delta:.4f}
            - Ingresos promedio: {income:.2f}
            - Score crediticio promedio: {score:.2f}
            - Saldo promedio: {saldo:.2f}
            - Uso de crédito promedio: {uso:.2f}
            """
        )

    return "\n".join(lines)


def build_variable_explanations() -> str:
    # Crea una sección textual explicando cada variable disponible en el dataset.
    # Esta sección proporciona contexto semántico para el RAG.
    var_dict = build_variable_dictionary()
    lines = ["DESCRIPCIÓN DE VARIABLES FINANCIERAS\n"]
    for var, desc in var_dict.items():
        lines.append(f"- {var}: {desc}")
    return "\n".join(lines)


def build_scenario_explanation_snippet() -> str:
    # Construye un texto descriptivo con las métricas del escenario más reciente.
    # Sirve para que el motor RAG explique conceptualmente el efecto del escenario.
    metrics = compute_scenario_metrics()
    if metrics is None:
        return "Aún no se ha simulado un escenario, no hay métricas de impacto disponibles."

    txt = f"""
    RESUMEN DEL ESCENARIO SIMULADO

    - Probabilidad media base: {metrics["mean_base"]:.4f}
    - Probabilidad media bajo escenario: {metrics["mean_scenario"]:.4f}
    - Delta promedio: {metrics["mean_delta"]:.4f}
    - Severidad del escenario: {metrics["severity"]}
    - Cluster más afectado: {metrics["cluster_worst"]} (delta promedio {metrics["cluster_worst_delta"]:.4f})
    - Top 5 clientes más afectados: {metrics["top5"]}

    Estas métricas resumen el impacto del escenario simulado sobre el riesgo de impago del portafolio.
    """
    return txt


def build_financial_knowledge_corpus() -> list[str]:
    # Reúne en una lista todos los textos que formarán el corpus del RAG.
    # Combina resúmenes, perfiles, explicaciones y métricas del escenario actual.
    df = load_latest_data()
    if df is None:
        return ["No hay datos cargados actualmente en el portafolio."]

    docs = []

    docs.append(build_portfolio_overview(df))
    docs.append(build_cluster_profiles(df))
    docs.append(build_variable_explanations())
    docs.append(build_scenario_explanation_snippet())

    return docs
