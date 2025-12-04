import streamlit as st

from api.executive_report import generate_executive_report

st.set_page_config(page_title="Reporte Ejecutivo de Riesgo", layout="wide")

st.title("Reporte Ejecutivo de Riesgo de Crédito")

st.markdown(
    """
Esta sección genera un reporte automático a nivel ejecutivo,
basado en el último escenario simulado sobre la cartera.
"""
)

if st.button("Generar reporte ejecutivo"):
    with st.spinner("Generando reporte con el escenario actual..."):
        try:
            report_text = generate_executive_report()
            st.success("Reporte generado.")
            st.markdown(report_text)
        except Exception as e:
            st.error(f"No fue posible generar el reporte: {e}")
