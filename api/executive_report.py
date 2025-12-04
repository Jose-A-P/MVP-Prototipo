import pandas as pd
from langchain_ollama import OllamaLLM

from api.digital_twin_tools import load_latest_data
from api.scenario_explainer import compute_scenario_metrics
from api.dataset_query_tools import (
    cluster_stats,
    top_n_by_delta,
)


def build_executive_context():
    """
    Construye el contexto numérico resumido para el reporte ejecutivo.
    """
    df = load_latest_data()
    if df is None:
        raise ValueError("No hay datos cargados. Ejecuta primero un escenario.")

    metrics = compute_scenario_metrics()
    if metrics is None:
        raise ValueError("No hay métricas de escenario. Ejecuta primero un escenario.")

    # Riesgo por cluster
    cl_stats = cluster_stats(df)

    # Top 10 clientes más afectados por delta_prob
    if "delta_prob" not in df.columns:
        df["delta_prob"] = df["prob_scenario"] - df["prob_base"]

    top_clients = top_n_by_delta(df, n=10)

    ctx = {
        "metrics": metrics,
        "cluster_stats": cl_stats.to_dict(orient="index"),
        "top_clients": top_clients.to_dict(orient="records"),
    }
    return ctx


def generate_executive_report() -> str:
    """
    Genera un reporte ejecutivo en texto largo, usando LLM + contexto numérico.
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
    - 2–3 párrafos explicando el impacto general del escenario sobre el portafolio.

    2. DESCRIPCIÓN DEL ESCENARIO
    - Explica qué tipo de shock representa, en términos de riesgo de crédito.

    3. IMPACTO GLOBAL EN EL PORTAFOLIO
    - Comenta la variación en probabilidad media de impago y qué implica.
    - Menciona la severidad (suave / moderado / severo) y su interpretación.

    4. ANÁLISIS POR CLUSTER
    - Indica qué cluster resulta más afectado y por qué podría ser más sensible.
    - Compara brevemente clusters menos afectados vs más afectados.

    5. CLIENTES CRÍTICOS
    - Explica el rol del top 10 de clientes en el deterioro agregado.
    - Sugiere qué tipo de acción podría tomarse sobre estos clientes (monitoreo, reestructuras, etc.).

    6. CONCLUSIONES Y RECOMENDACIONES
    - Señala 3–5 bullets de acciones sugeridas para gestión de riesgo (monitoreo, límites, alertas, stress testing adicional, etc.).

    Usa un tono profesional, orientado a comité, claro y conciso,
    sin inventar números que no estén en el contexto.
    """

    return llm.invoke(prompt)
