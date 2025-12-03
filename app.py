import streamlit as st

st.set_page_config(page_title="Digital Twin Financiero", layout="wide")

st.title("Digital Twin Financiero – Navegación")

st.write("""
Selecciona una página desde el menú lateral:
- **Chatbot** para interactuar con el modelo y ejecutar escenarios.
- **Densidad de Probabilidades** para visualizar cambios en riesgo.
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