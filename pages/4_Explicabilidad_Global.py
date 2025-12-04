import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from api.shap_explainer import shap_global

st.set_page_config(layout="wide")
st.title("Explicabilidad Global del Modelo de Impago (SHAP)")

# Cargar datos SHAP
try:
    explainer, shap_values, X, df, feature_names = shap_global()
except Exception as e:
    st.error(f"No se pudo generar explicabilidad: {str(e)}")
    st.stop()

st.markdown("""
Esta sección muestra la explicabilidad **global** del modelo de impago utilizando:
- Importancia global de características
- Comportamiento de variables en el portafolio
- Sensibilidad del modelo
- Relaciones entre variables y contribuciones SHAP
""")

# GLOBAL FEATURE IMPORTANCE (BAR PLOT)

st.subheader("Importancia Global de las Variables")

fig_bar, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig_bar, use_container_width=True)

st.markdown("""
El gráfico anterior muestra las variables con **mayor contribución absoluta** 
a la probabilidad de impago, agregadas a nivel global.
""")

st.divider()

# BEESWARM SUMMARY PLOT

st.subheader("SHAP Beeswarm Summary Plot")

fig_bee, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig_bee, use_container_width=True)

st.markdown("""
El **beeswarm plot** revela:
- La **importancia global** de cada variable.
- La **dirección** del efecto (alto = rojo, bajo = azul).
- La **dispersión** de contribuciones SHAP en el portafolio.
""")

st.divider()

# DEPENDENCE PLOT (VARIABLE ESPECÍFICA)
st.subheader("SHAP Dependence Plot (Variable Específica)")

variable = st.selectbox(
    "Selecciona la variable para analizar:",
    feature_names
)

# Limpiar figura actual
plt.clf()
plt.figure(figsize=(10, 6))

shap.dependence_plot(
    variable,
    shap_values,
    X,
    interaction_index=None,
    show=False
)

st.pyplot(plt.gcf(), use_container_width=True)

st.markdown(f"""
El gráfico muestra cómo **{variable}** afecta la probabilidad de impago según SHAP.
- Cada punto es un cliente.
- Eje X: valor de la variable.
- Eje Y: contribución SHAP (impacto en el riesgo).
""")

# DESCRIPCIÓN AUTOMÁTICA DEL MODELO
st.subheader("Interpretación Automática del Modelo")

# Promedios absolutos SHAP
mean_abs = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs)[::-1]

top_vars = [feature_names[i] for i in sorted_idx[:5]]
txt = "### Factores Globales que Explican el Riesgo\n"

txt += "\n**Las 5 variables más importantes globalmente:**\n"
for v in top_vars:
    txt += f"- **{v}**\n"

txt += """

Estas variables tienen mayor influencia en el riesgo de impago según el modelo 
entrenado y pueden ser utilizadas para estrategias de monitoreo, 
alertas tempranas y segmentación de riesgo.
"""

st.markdown(txt)
