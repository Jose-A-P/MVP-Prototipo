import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM   # si funciona
import os

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
    llm = OllamaLLM(model="gemma:2b")  # se puede usar "mistral", "gemma:2b", etc.

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
