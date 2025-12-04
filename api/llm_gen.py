import pandas as pd
import os
import re

from langchain_ollama import OllamaLLM

# === RAG PIPELINE
from api.rag_chat import get_rag_answer

# === Escenarios y Digital Twin
from api.digital_twin_tools import (
    chatbot_execute_action,
    load_latest_data
)

# === SHAP
from api.shap_explainer import shap_for_client

# === Dataset Query Tools
from api.dataset_query_tools import (
    parse_dataset_query,
    top_n_by_delta,
    worst_clients,
    cluster_stats,
    get_client_info,
    filter_by_risk,
    extract_top_n,
    extract_cluster,
    extract_threshold,
    apply_filters
)

from api.scenario_manager import (
        list_scenarios,
        load_scenario,
        compare_scenarios
    )

from api.scenario_explainer import compute_scenario_metrics

from api.executive_report import generate_executive_report

# ============================================================
# Función auxiliar para explicar escenarios
# ============================================================
def generate_scenario_explanation():
    metrics = compute_scenario_metrics()
    if metrics is None:
        return "Aún no existe un escenario simulado para analizar."

    llm = OllamaLLM(model="mistral")

    prompt = f"""
    Eres un analista financiero especializado en riesgo de crédito.
    Explica los resultados del escenario simulado utilizando las siguientes métricas:

    - Probabilidad media base: {metrics["mean_base"]:.4f}
    - Probabilidad media bajo escenario: {metrics["mean_scenario"]:.4f}
    - Aumento promedio (delta): {metrics["mean_delta"]:.4f}
    - Severidad del impacto: {metrics["severity"]}

    - Cluster más afectado: {metrics["cluster_worst"]}
      (delta promedio = {metrics["cluster_worst_delta"]:.4f})

    - Top 5 clientes más afectados:
      {metrics["top5"]}

    Elabora un análisis estructurado con:
    * Descripción general del escenario
    * Identificación de deterioros relevantes
    * Análisis por cluster
    * Implicaciones para riesgo crediticio
    * Riesgos emergentes
    * Conclusiones ejecutivas
    """

    return llm.invoke(prompt)


# ============================================================
# PATRONES para activar RAG (preguntas conceptuales financieras)
# ============================================================
FINANCIAL_CONCEPTUAL_PATTERNS = [
    "riesgo",
    "cluster concentra",
    "cluster más afectado",
    "cual cluster",
    "impacto del escenario",
    "resumen escenario",
    "explica el escenario",
    "sensibilidad",
    "concentra mayor",
    "deterioro",
    "afectación del escenario",
    "exposición",
    "bajo el escenario",
]


def is_conceptual_financial_question(text):
    # Si la pregunta menciona explícitamente columnas NO ES RAG
    tabular_terms = [
        "prob_base", "prob_scenario", "delta_prob",
        "saldo", "ingreso", "cluster", "cliente", "id_"
    ]
    if any(t in text for t in tabular_terms):
        return False

    if re.search(r"\d+(\.\d+)?", text):
        return False

    # Pero permitimos preguntas conceptuales
    conceptual_terms = [
        "explica",
        "riesgo",
        "impacto",
        "interpretación",
        "tendencia",
        "concentración de riesgo",
        "análisis del portafolio",
        "qué significa",
        "interpretar",
        "cómo afecta",
        "conclusión",
    ]

    return any(t in text for t in conceptual_terms)


# ============================================================
# Detectar preguntas SHAP
# ============================================================
def detect_shap_query(text):
    return "shap" in text or "explica cliente" in text or "importancia cliente" in text


def extract_client_id(text):
    text = text.upper()
    match = re.search(r"C\d{3,6}", text)
    return match.group(0) if match else None


# ============================================================
#  Detectar escenarios
# ============================================================
def detect_scenario_query(text):
    return any(k in text for k in ["ingreso", "tasa", "impago", "mora", "combinado"])


# ============================================================
# Motor de respuesta — ROUTER PRINCIPAL
# ============================================================
def generate_chat_response(prompt: str) -> str:
    text = prompt.lower()
    df = load_latest_data()

    # ============================================================
    # ESCENARIOS (si detecta escenario es prioridad absoluta)
    # ============================================================
    if detect_scenario_query(text):
        result = chatbot_execute_action(prompt)
        if result:
            return result

    # ============================================================
    # SHAP: explicabilidad por cliente
    # ============================================================
    if detect_shap_query(text):
        client_id = extract_client_id(text)
        if not client_id:
            return "Debes indicar un ID de cliente, por ejemplo C01234."
        shap_df, txt = shap_for_client(client_id)
        return txt

    # ============================================================
    # TABULAR QUERIES (Dataset Query Router)
    # ============================================================
    if df is not None:
        parsed = parse_dataset_query(prompt, df)

        if parsed["intent"] is not None:

            # ----------------------------
            # MEDIA
            # ----------------------------
            if parsed["intent"] == "mean":
                col = parsed["columns"][0] if parsed["columns"] else None
                if col:
                    return f"La media de **{col}** es **{df[col].mean():.4f}**."
                return "¿De qué columna deseas obtener la media?"

            # ----------------------------
            # MAX
            # ----------------------------
            if parsed["intent"] == "max":
                col = parsed["columns"][0]
                row = df.loc[df[col].idxmax()]
                return f"Cliente con máximo {col}: {row['ID_cliente']} (valor: {row[col]:.4f})"

            # ----------------------------
            # MIN
            # ----------------------------
            if parsed["intent"] == "min":
                col = parsed["columns"][0]
                row = df.loc[df[col].idxmin()]
                return f"Cliente con mínimo {col}: {row['ID_cliente']} (valor: {row[col]:.4f})"

            # ----------------------------
            # DESCRIBE
            # ----------------------------
            if parsed["intent"] == "describe":
                col = parsed["columns"][0]
                return f"```\n{df[col].describe()}\n```"

            # ----------------------------
            # TOP DELTA
            # ----------------------------
            if parsed["intent"] == "top_delta":
                n = extract_top_n(prompt) or 5
                cluster = extract_cluster(prompt)
                column, threshold = extract_threshold(prompt, df)

                filtered = apply_filters(df, cluster, column, threshold)

                if "delta_prob" not in filtered:
                    filtered["delta_prob"] = filtered["prob_scenario"] - filtered["prob_base"]

                top = filtered.nlargest(n, "delta_prob")

                return f"```\n{top.to_string()}\n```"

            # ----------------------------
            # INFORMACIÓN CLIENTE
            # ----------------------------
            if parsed["intent"] == "client_info":
                info = get_client_info(df, parsed["client_id"])
                return f"```\n{info}\n```"

            # ----------------------------
            # RIESGO POR CLUSTER
            # ----------------------------
            if parsed["intent"] == "cluster_risk":
                stats = cluster_stats(df)
                return f"```\n{stats}\n```"
    
    # ============================================================
    # LISTAR ESCENARIOS
    # ============================================================
    if "lista escenarios" in text or "listar escenarios" in text:
        scenarios = list_scenarios()
        if not scenarios:
            return "Aún no hay escenarios guardados."
        
        out = "**Escenarios disponibles:**\n"
        for s in scenarios:
            out += (
                f"- Escenario {s['scenario_id']:03d} | "
                f"{s['timestamp']} | "
                f"tipo: {s.get('type', 'N/A')}\n"
            )
        return out

    # ============================================================
    # CARGAR ESCENARIOS
    # ============================================================
    match = re.search(r"carga[r]? escenario (\d+)", text)
    if match:
        sid = int(match.group(1))
        loaded = load_scenario(sid)
        if loaded is None:
            return f"No existe el escenario {sid:03d}."
        
        df_comb, df_imp, meta = loaded
        return (
            f"**Escenario {sid:03d} cargado correctamente.**\n\n"
            f"Metadata:\n```\n{meta}\n```"
        )

    # ============================================================
    # COMPARAR ESCENARIOS
    # ============================================================
    match = re.search(r"compara[r]? escenario (\d+) con (\d+)", text)
    if match:
        s1 = int(match.group(1))
        s2 = int(match.group(2))
        comp = compare_scenarios(s1, s2)

        if comp is None:
            return "No se pudo comparar los escenarios."

        top = comp["top_risk_increase"].head().to_string()

        return (
            f"**Comparación entre escenarios {s1:03d} y {s2:03d}:**\n\n"
            f"- Δ medio S{s1:03d}: **{comp['mean_delta_1']:.4f}**\n"
            f"- Δ medio S{s2:03d}: **{comp['mean_delta_2']:.4f}**\n"
            f"- Diferencia: **{comp['diff_mean']:.4f}**\n\n"
            f"Clientes con mayor deterioro adicional en el escenario {s2:03d}:\n"
            f"```\n{top}\n```"
        )


    # ============================================================
    # REPORTE EJECUTIVO (comandos tipo "reporte ejecutivo")
    # ============================================================
    if any(
        k in text
        for k in [
            "reporte ejecutivo",
            "informe ejecutivo",
            "reporte para comité",
            "informe para comité",
            "reporte de riesgo",
        ]
    ):
        try:
            return generate_executive_report()
        except Exception as e:
            return f"No pude generar el reporte ejecutivo. Verifica que ya hayas corrido un escenario.\nDetalle técnico: {e}"
    # ============================================================
    # RAG — preguntas conceptuales
    # ============================================================
    if is_conceptual_financial_question(text):
        try:
            answer = get_rag_answer(prompt)
            return answer
        except Exception as e:
            return f"Error en RAG: {e}"

    # ============================================================
    # FALLBACK → LLM directo
    # ============================================================
    llm = OllamaLLM(model="mistral")
    return llm.invoke(prompt)