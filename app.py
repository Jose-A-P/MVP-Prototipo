import streamlit as st

st.set_page_config(page_title="Digital Twin Financiero", layout="wide")

import streamlit as st

st.markdown("""
# Digital Twin de Riesgo Crediticio — Panel Principal

Bienvenido al Digital Twin Financiero, una plataforma interactiva diseñada para simular escenarios de riesgo crediticio, analizar su impacto sobre la cartera de clientes y generar información accionable para la toma de decisiones en banca.

El sistema integra modelos de machine learning, simulación de escenarios, RAG financiero, explicabilidad SHAP y visualizaciones dinámicas para ofrecer una visión completa del riesgo tanto a nivel global como individual.

---

## Funciones Principales

Esta aplicación permite:

### Simular escenarios de estrés
Modifique ingresos, tasas de interés o comportamiento de pago para observar cómo cambia la probabilidad de impago de los clientes.

### Consultar información del dataset
A través del chatbot puede solicitar:
- Promedios, mínimos, máximos  
- Clientes más afectados  
- Riesgo por cluster  
- Información individual  
- Filtrado por umbrales o características específicas  

### Visualizar resultados
Incluye distribuciones, densidades por cluster, gráficos comparativos y dispersión entre probabilidades base y simulada.

### Explicar modelos con SHAP
Analice la contribución de cada variable al riesgo de un cliente específico o a nivel global.

### Recibir análisis conceptual mediante RAG financiero
Obtenga explicaciones descriptivas y ejecutivas basadas en el escenario simulado, el portafolio y los clusters.

---

## Descripción de las Páginas

### 1. Chatbot Inteligente
Página de interacción principal.  
Permite:
- Ejecutar escenarios con lenguaje natural  
- Consultar datos del dataset  
- Solicitar explicaciones de riesgo  
- Recibir reportes automáticos del escenario  
- Acceder al motor RAG para preguntas conceptuales  

Ejemplos:
- "Simula un escenario donde disminuye ingreso 10%, se presenta mora 0.1 e incrementa tasa 1%"  
- "Dame los 5 clientes más afectados"  
- "Explica que significa el aumento de tasa de interes en el escenario anterior"

---

### 2. Densidad de Probabilidades
Visualiza cómo cambian las distribuciones de riesgo bajo el escenario.

Incluye:
- KDE global  
- KDE por cluster  
- KDE por delta_prob  
- Scatter interactivo  

Permite detectar desplazamientos de riesgo, sensibilidad de clusters y concentración de deterioros.

---

### 3. Explicabilidad SHAP (Cliente individual)
Muestra la contribución de las variables al riesgo de un cliente.

Incluye:
- Ranking de variables  
- Explicación textual generada por IA  
- Valores SHAP individuales  

Ejemplo:
- "Explica cliente C01234"

---

### 4. Explicabilidad Global
Muestra la importancia general de las variables en el modelo.

Incluye:
- Feature importance  
- SHAP summary plot  
- Interpretaciones narrativas  

Revela los factores estructurales del riesgo en la cartera.

---

### 5. Motor de Alertas Tempranas
Clasifica clientes en niveles de alerta según:
- Probabilidad base  
- Probabilidad bajo escenario  
- Deterioro (delta_prob)  
- Umbrales configurables  

Permite identificar deterioros significativos y clusters críticos.

---

### 6. Reportes Ejecutivos (RAG Financiero)
Genera análisis narrativos basados únicamente en:
- El escenario simulado  
- Las métricas del portafolio  
- Los clusters  
- El diccionario de variables financieras  

Ideal para presentaciones e informes ejecutivos.

---

## Tecnologías Integradas

- Machine Learning (Scikit-learn)  
- Simulación paramétrica de escenarios  
- Explainable AI (SHAP)  
- Retrieval Augmented Generation (RAG)  
- ChromaDB para almacenamiento vectorial  
- Ollama LLM (Mistral)  
- Streamlit para visualización

---

## Futuras Mejoras Sugeridas

- Panel ejecutivo consolidado  
- Comparación histórica de escenarios  
- Descarga de reportes PDF  
- Análisis contrafactuales  
- Integración con bases transaccionales reales  
""")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# # generar reporte con llm
# from api.llm_gen import generate_chat_response
# from api.digital_twin_tools import load_latest_data

# # STREAMLIT UI
# st.set_page_config(page_title="Digital Twin Financiero", layout="wide")

# # === RUTAS COHERENTES CON digital_twin.py ===
# BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

# # === RUTAS COHERENTES CON digital_twin.py ===
# BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

# CSV_PATH = os.path.join(BASE_DIR, "clientes_simulados_combined.csv")  # <-- usar este
# MODEL_PATH = os.path.join(BASE_DIR, "modelo_logistico_impago.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler_modelo.pkl")
# CLIENTES_COMBINED_PATH = os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv")

# # CARGAR DATA
# #df = pd.read_csv(CLIENTES_COMBINED_PATH)
# df = load_latest_data()

# st.title("Digital Twin — Chatbot")

# user_input = st.chat_input("Haz una pregunta sobre los clientes:")

# if user_input:
#     with st.chat_message("user"):
#         st.write(user_input)

#     with st.chat_message("assistant"):
#         response = generate_chat_response(user_input)
#         st.write(response)