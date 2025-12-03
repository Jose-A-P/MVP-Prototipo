import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM   # si funciona
import os
from api.dataset_query_tools import (
    detect_query_type,
    get_mean, get_max, get_min, describe_column,
    top_n_by_delta, worst_clients, cluster_stats,
    get_client_info, filter_by_risk, get_portfolio_risk_summary
)
from api.digital_twin_tools import load_latest_data

'''
llm = OllamaLLM(
    model="llama2",
    base_url=os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
)
'''

def generate_summary(df):
    # Calcular métricas agregadas
    mean_base = df["prob_base"].mean()
    mean_scenario = df["prob_scenario"].mean()
    delta = mean_scenario - mean_base

    # Agrupar por cluster usando columnas correctas
    cluster_summary = df.groupby("cluster_kmeans")[["prob_base","prob_scenario","delta_prob"]].mean()

    # Configurar LLM local con Ollama
    llm = OllamaLLM(model="mistral")  # se puede usar "mistral", "gemma:2b", etc.

    # Prompt para informe ejecutivo
    template = """
    Eres un analista financiero. Resume los resultados del Digital Twin Financiero en lenguaje ejecutivo.
    Datos:
    - Probabilidad promedio base: {mean_base:.3f}
    - Probabilidad promedio escenario: {mean_scenario:.3f}
    - Cambio promedio: {delta:.3f}
    - Impacto por cluster: {cluster_summary}

    Genera un informe claro para gerencia, destacando riesgos y recomendaciones.
    """

    prompt = PromptTemplate(
        input_variables=["mean_base","mean_scenario","delta","cluster_summary"],
        template=template
    )

    # Ejecutar directamente con Ollama
    summary = llm.invoke(prompt.format(
        mean_base=mean_base,
        mean_scenario=mean_scenario,
        delta=delta,
        cluster_summary=cluster_summary.to_dict()
    ))

    return summary

def generate_chat_response(prompt: str, df_context: pd.DataFrame) -> str:
    # Configurar LLM local con Ollama
    llm = OllamaLLM(model="mistral")

    # Resumir el DataFrame para dar contexto
    sample = df_context.to_string()
    stats = df_context.describe().to_string()

    # 1. Intento identificar pregunta sobre dataset
    query_type = detect_query_type(prompt)

    df = load_latest_data()

    if df is not None and query_type is not None:
        prompt_l = prompt.lower()

        if query_type == "mean":
            for col in df.columns:
                if col in prompt_l:
                    return f"La media de {col} es {df[col].mean():.4f}"

        if query_type == "max":
            for col in df.columns:
                if col in prompt_l:
                    row = df.loc[df[col].idxmax()]
                    return f"Valor máximo en {col}: {row[col]:.4f}\nCliente: {row['ID_cliente']}"

        if query_type == "min":
            for col in df.columns:
                if col in prompt_l:
                    row = df.loc[df[col].idxmin()]
                    return f"Valor mínimo en {col}: {row[col]:.4f}\nCliente: {row['ID_cliente']}"

        if query_type == "describe":
            for col in df.columns:
                if col in prompt_l:
                    return str(df[col].describe())

        if query_type == "top_delta":
            result = top_n_by_delta(df, n=10)
            return result.to_string()

        if query_type == "cluster_risk":
            result = cluster_stats(df)
            return result.to_string()

        if query_type == "risk_filter":
            riesgo = filter_by_risk(df, threshold=0.4)
            return riesgo.to_string()

        if query_type == "client_info":
            import re
            match = re.search(r"C\d{5}", prompt)
            if match:
                client_id = match.group(0)
                return str(get_client_info(df, client_id))

    full_prompt = f"""
    Usa la siguiente información de clientes como contexto:

    Muestra de datos:
    {sample}

    Estadísticas generales:
    {stats}

    Pregunta del usuario: {prompt}

    Responde basándote en estos datos teniendo el rol de un analista financiero .
    """

    return llm.invoke(full_prompt)
