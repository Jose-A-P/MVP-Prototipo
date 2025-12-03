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
    filter_by_risk
)


# ============================================================
# FUNCIÓN PRINCIPAL DEL CHATBOT
# ============================================================

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
                top = top_n_by_delta(df, n=10)
                return f"Top 10 clientes con mayor deterioro:\n\n```\n{top.to_string()}\n```"

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


# import pandas as pd
# from langchain.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM
# from api.digital_twin_tools import chatbot_execute_action, load_latest_data
# from api.rag_chat import build_rag_pipeline
# from api.shap_explainer import shap_for_client
# import re
# from api.dataset_query_tools import (
#     parse_dataset_query,
#     get_mean, get_max, get_min, describe_column,
#     top_n_by_delta, worst_clients, cluster_stats,
#     get_client_info, filter_by_risk, get_portfolio_risk_summary
# )
# from api.digital_twin_tools import load_latest_data

# # Helper: construir prompt

# FINANCIAL_ANALYST_PROMPT = """
# Eres un analista financiero experto que usa los datos del cliente 
# y la información recuperada del RAG para responder preguntas.

# Contexto relevante recuperado:
# {context}

# Pregunta del usuario:
# {question}

# Instrucciones:
# - Si la pregunta requiere ejecutar un escenario (caída de ingresos, aumento de tasa, peor comportamiento, escenario combinado),
#   debes indicarlo y el sistema ejecutará la función correspondiente.
# - Si el usuario pregunta datos numéricos, resúmenes, tendencias o patrones, analiza el contexto.
# - Si la pregunta no puede responderse con el contexto, explica por qué.

# Responde de forma clara, con razonamiento financiero.
# """

# # Función principal: combinación de TOOL EXECUTION + RAG

# def generate_chat_response(prompt: str) -> str:
#     """
#     1. Primero detecta si el usuario quiere ejecutar un escenario del digital twin.
#          - Si  → ejecuta la función → actualiza dataset → responde.
#     2. Si no hay acción → usa RAG para responder basado en datos.
#     """

#     # Intentar ejecutar una acción del Digital Twin
#     action_result = chatbot_execute_action(prompt)

#     if action_result is not None:
#         # El chatbot ejecutó un escenario real
#         return f"{action_result}\n\nLos datos fueron actualizados."

#     if "explica" in prompt.lower() or "shap" in prompt.lower():
#         match = re.search(r"C\d{5}", prompt)
#         if match:
#             client_id = match.group(0)
#             shap_df, summary = shap_for_client(client_id)
#             if shap_df is None:
#                 return summary  # mensaje de error
#             return summary
        
#     query_type = detect_query_type(prompt)

#     # Cargar dataset actualizado para el RAG
#     df = load_latest_data()
#     if df is not None and query_type is not None:
#         prompt_l = prompt.lower()

#         if query_type == "mean":
#             for col in df.columns:
#                 if col in prompt_l:
#                     return f"La media de {col} es {df[col].mean():.4f}"

#         if query_type == "max":
#             for col in df.columns:
#                 if col in prompt_l:
#                     row = df.loc[df[col].idxmax()]
#                     return f"Valor máximo en {col}: {row[col]:.4f}\nCliente: {row['ID_cliente']}"

#         if query_type == "min":
#             for col in df.columns:
#                 if col in prompt_l:
#                     row = df.loc[df[col].idxmin()]
#                     return f"Valor mínimo en {col}: {row[col]:.4f}\nCliente: {row['ID_cliente']}"

#         if query_type == "describe":
#             for col in df.columns:
#                 if col in prompt_l:
#                     return str(df[col].describe())

#         if query_type == "top_delta":
#             result = top_n_by_delta(df, n=10)
#             return result.to_string()

#         if query_type == "cluster_risk":
#             result = cluster_stats(df)
#             return result.to_string()

#         if query_type == "risk_filter":
#             riesgo = filter_by_risk(df, threshold=0.4)
#             return riesgo.to_string()

#         if query_type == "client_info":
#             import re
#             match = re.search(r"C\d{5}", prompt)
#             if match:
#                 client_id = match.group(0)
#                 return str(get_client_info(df, client_id))
#     if df is None:
#         return "No existe dataset combinado. Genera un escenario primero."
    
#     # Construir pipeline RAG (Retriever + LLM)
#     rag_chain = build_rag_pipeline()

#     # Ejecutar RAG
#     try:
#         response = rag_chain.invoke({"query": prompt})
#         return response["result"]
#     except Exception as e:
#         return f"Error en el RAG: {e}"