import threading
import webview
import subprocess
import time
import os

STREAMLIT_APP = "app.py"

def start_streamlit():
    """
    Ejecuta Streamlit en un hilo separado
    """
    cmd = f"streamlit run {STREAMLIT_APP} --server.headless true --server.port 8501"
    subprocess.Popen(cmd, shell=True)

def create_window():
    """
    Crea una ventana PyWebview cargando http://localhost:8501
    """
    # Esperar a que streamlit levante
    time.sleep(2)

    webview.create_window(
        "Digital Twin Financiero",
        "http://localhost:8501",
        width=1400,
        height=900,
        resizable=True
    )
    webview.start()

if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()
    create_window()
