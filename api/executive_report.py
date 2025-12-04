import pandas as pd
from langchain_ollama import OllamaLLM

from api.digital_twin_tools import load_latest_data
from api.scenario_explainer import compute_scenario_metrics
from api.dataset_query_tools import (
    cluster_stats,
    top_n_by_delta,
)

"""
Este módulo genera reportes ejecutivos basados en los resultados del Digital Twin.

Este reporte se utiliza en la interfaz Streamlit para generar un documento
que sintetice los hallazgos del escenario de estrés.
"""


# ============================================================
# Construcción del contexto numérico usado por el LLM
# ============================================================

def build_executive_context():
    """
    Construye un diccionario de contexto para el reporte ejecutivo.

    El contexto incluye:
    - Métricas globales del escenario simulado, calculadas por compute_scenario_metrics.
    - Estadísticas agrupadas por cluster, generadas con cluster_stats.
    - Top 10 clientes más afectados según delta_prob.

    Este contexto es posteriormente inyectado en el prompt del LLM
    para redactar el reporte de manera estructurada.
    """

    df = load_latest_data()
    if df is None:
        raise ValueError("No hay datos cargados. Ejecuta primero un escenario.")

    metrics = compute_scenario_metrics()
    if metrics is None:
        raise ValueError("No hay métricas de escenario. Ejecuta primero un escenario.")

    cl_stats = cluster_stats(df)

    if "delta_prob" not in df.columns:
        df["delta_prob"] = df["prob_scenario"] - df["prob_base"]

    top_clients = top_n_by_delta(df, n=10)

    ctx = {
        "metrics": metrics,
        "cluster_stats": cl_stats.to_dict(orient="index"),
        "top_clients": top_clients.to_dict(orient="records"),
    }
    return ctx


# ============================================================
# Generación del reporte ejecutivo
# ============================================================

def generate_executive_report() -> str:
    """
    Genera un reporte ejecutivo usando un modelo LLM (Mistral).

    El flujo consiste en:
    1. Construir un contexto cuantitativo mediante build_executive_context.
    2. Inyectar estas métricas en un prompt estructurado.
    3. Solicitar al modelo redactar un informe profesional que siga
       un esquema estándar de comité de riesgo:

       - Resumen ejecutivo
       - Descripción del escenario
       - Impacto global
       - Análisis por cluster
       - Clientes críticos
       - Conclusiones y recomendaciones

    Este reporte es uno de los productos principales del Digital Twin,
    ya que permite comunicar hallazgos de forma precisa a tomadores de decisión.
    """

    ctx = build_executive_context()

    metrics = ctx["metrics"]
    cl_stats = ctx["cluster_stats"]
    top_clients = ctx["top_clients"]

    llm = OllamaLLM(model="mistral")

    prompt = f"""
    Eres un analista senior de riesgo de crédito en un banco.
    Debes redactar un REPORTE EJECUTIVO para comité de riesgo,
    a partir de los siguientes datos del escenario simulado.

    MÉTRICAS GLOBALES DEL ESCENARIO
    - Probabilidad media base de impago: {metrics["mean_base"]:.4f}
    - Probabilidad media bajo escenario: {metrics["mean_scenario"]:.4f}
    - Delta promedio de probabilidad de impago: {metrics["mean_delta"]:.4f}
    - Severidad del escenario: {metrics["severity"]}

    CLUSTER MÁS AFECTADO:
    - Cluster: {metrics["cluster_worst"]}
    - Delta promedio en ese cluster: {metrics["cluster_worst_delta"]:.4f}

    RIESGO POR CLUSTER (resumen por cluster_kmeans):
    {cl_stats}

    TOP 10 CLIENTES MÁS AFECTADOS (por delta_prob):
    {top_clients}

    Redacta un reporte ejecutivo en ESPAÑOL, con el siguiente esquema:

    1. RESUMEN EJECUTIVO
    Explica en varios párrafos la visión general del impacto del escenario.

    2. DESCRIPCIÓN DEL ESCENARIO
    Describe la naturaleza del shock y sus implicaciones en riesgo de crédito.

    3. IMPACTO GLOBAL EN EL PORTAFOLIO
    Analiza los cambios agregados en probabilidad de impago y su interpretación.

    4. ANÁLISIS POR CLUSTER
    Evalúa cuáles grupos de clientes son más sensibles y por qué.

    5. CLIENTES CRÍTICOS
    Explica la relevancia del top 10 de clientes en el deterioro del portafolio.

    6. CONCLUSIONES Y RECOMENDACIONES
    Propón recomendaciones de gestión de riesgo basadas en los hallazgos.

    Usa lenguaje profesional, preciso y orientado a comité,
    sin inventar datos que no estén en el contexto.
    """

    return llm.invoke(prompt)
