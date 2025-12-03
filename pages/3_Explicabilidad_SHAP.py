import streamlit as st
import shap
import pandas as pd
from api.shap_explainer import shap_for_client

st.title("Explicabilidad SHAP del Modelo de Impago")

client_id = st.text_input("Ingresa un ID de cliente (ej. C01234)")

if st.button("Explicar"):
    shap_df, txt = shap_for_client(client_id)
    if shap_df is None:
        st.error(txt)
    else:
        st.markdown(txt)

        st.subheader("Contribuciones SHAP")
        st.dataframe(shap_df, use_container_width=True)
