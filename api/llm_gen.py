import pandas as pd
import os

from langchain_ollama import OllamaLLM

# === RAG PIPELINE (si lo usas)
from api.rag_chat import build_rag_pipeline

# === Escenarios y Digital Twin
from api.digital_twin_tools import (
    chatbot_execute_action,
    load_latest_data
)

# === SHAP
from api.shap_explainer import shap_for_client

# === Dataset Query Tools (router inteligente)
from api.dataset_query_tools import (
    parse_dataset_query,
    top_n_by_delta,
    worst_clients,
    cluster_stats,
    get_client_info,
    filter_by_risk,
    extract_top_n,
    extract_top_n, 
    extract_cluster, 
    extract_threshold, 
    apply_filters
)

from api.scenario_explainer import compute_scenario_metrics

# FUNCION AUXILIAR DEL CHATBOT
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

    Elabora un análisis narrativo con:
    * Explicación general del escenario
    * Identificación de deterioros relevantes
    * Implicaciones para riesgo crediticio
    * Impacto por cluster
    * Riesgos emergentes
    * Conclusiones ejecutivas
    """

    return llm.invoke(prompt)

# FUNCIÓN PRINCIPAL DEL CHATBOT
def generate_chat_response(prompt: str) -> str:
    prompt_l = prompt.lower()

    # ============================================================
    # ROUTER INTELIGENTE PARA CONSULTAS DEL DATASET (PRIORIDAD ALTA)
    # ============================================================
    df = load_latest_data()

    if df is not None:
        parsed = parse_dataset_query(prompt, df)

        if parsed["intent"] is not None:
            # ----------------------------
            # MEDIA / PROMEDIO
            # ----------------------------
            if parsed["intent"] == "mean":
                if parsed["columns"]:
                    col = parsed["columns"][0]
                    return f"La media de **{col}** es **{df[col].mean():.4f}**."
                else:
                    return "¿De qué columna deseas conocer el promedio?"

            # ----------------------------
            # MÁXIMO
            # ----------------------------
            if parsed["intent"] == "max":
                if parsed["columns"]:
                    col = parsed["columns"][0]
                    row = df.loc[df[col].idxmax()]
                    return (
                        f"Máximo valor en **{col}**: **{row[col]:.4f}**\n"
                        f"Cliente: **{row['ID_cliente']}**"
                    )
                return "¿Qué columna deseas consultar?"

            # ----------------------------
            # MÍNIMO
            # ----------------------------
            if parsed["intent"] == "min":
                if parsed["columns"]:
                    col = parsed["columns"][0]
                    row = df.loc[df[col].idxmin()]
                    return (
                        f"Mínimo valor en **{col}**: **{row[col]:.4f}**\n"
                        f"Cliente: **{row['ID_cliente']}**"
                    )
                return "¿Qué columna deseas consultar?"

            # ----------------------------
            # DESCRIBE
            # ----------------------------
            if parsed["intent"] == "describe":
                if parsed["columns"]:
                    col = parsed["columns"][0]
                    return f"Descripción estadística de **{col}**:\n\n```\n{df[col].describe()}\n```"
                return "¿Qué columna deseas describir?"

            # ----------------------------
            # TOP DELTA
            # ----------------------------
            if parsed["intent"] == "top_delta":
                n = extract_top_n(prompt) or 10

                # Extraer cluster si viene en la consulta
                cluster = extract_cluster(prompt)

                # Extraer posibles filtros tipo "mayor a 0.4"
                column, threshold = extract_threshold(prompt, df)

                # Aplicar filtros
                filtered_df = apply_filters(df, cluster=cluster, column=column, threshold=threshold)

                # Recalcular delta si no existe
                if "delta_prob" not in filtered_df.columns:
                    filtered_df["delta_prob"] = filtered_df["prob_scenario"] - filtered_df["prob_base"]

                # Tomar top N
                top = filtered_df.sort_values("delta_prob", ascending=False).head(n)

                # Construir texto descriptivo dinámico
                desc_parts = [f"Top {n} clientes con mayor deterioro"]
                if cluster is not None:
                    desc_parts.append(f"en el cluster {cluster}")
                if column and threshold:
                    desc_parts.append(f"con {column} >= {threshold}")

                desc = " ".join(desc_parts)

                return f"{desc}:\n\n```\n{top.to_string()}\n```"

            # ----------------------------
            # RIESGO POR CLUSTER
            # ----------------------------
            if parsed["intent"] == "cluster_risk":
                stats = cluster_stats(df)
                return f"Riesgo por cluster:\n\n```\n{stats.to_string()}\n```"

            # ----------------------------
            # FILTRO DE RIESGO
            # ----------------------------
            if parsed["intent"] == "risk_filter":
                filt = filter_by_risk(df, threshold=0.4)
                return f"Clientes con riesgo >= 0.4:\n\n```\n{filt.to_string()}\n```"

            # ----------------------------
            # INFORMACIÓN DE CLIENTE
            # ----------------------------
            if parsed["intent"] == "client_info":
                return str(get_client_info(df, parsed["client_id"]))

    # ============================================================
    # SHAP PARA UN CLIENTE
    # ============================================================
    if "shap" in prompt_l or "explica" in prompt_l:
        import re
        match = re.search(r"C\d{5}", prompt)
        if match:
            client_id = match.group(0)
            shap_df, txt = shap_for_client(client_id)
            if shap_df is None:
                return txt
            return txt  # resumen SHAP en texto

    # ============================================================
    # EJECUCIÓN DE ACCIONES (ESCENARIOS)
    # ============================================================
    action_result = chatbot_execute_action(prompt)
    if action_result is not None:
        return action_result

    # ============================================================
    # RAG (para interpretaciones textuales complejas)
    # ============================================================
    try:
        rag = build_rag_pipeline()
        chain = rag.get_qa_chain()
        return chain.invoke({"query": prompt})
    except:
        pass

    # ============================================================
    # FALLBACK: USAR LLM (mistral)
    # ============================================================
    llm = OllamaLLM(model="mistral")
    return llm.invoke(prompt)