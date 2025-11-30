FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    streamlit fastapi uvicorn \
    scikit-learn pandas numpy matplotlib seaborn joblib \
    langchain==0.2.14 langchain-core==0.2.34 \
    langchain-community==0.2.12 langchain-ollama==0.1.0 \
    ollama==0.1.9 mlflow==2.17.0 pydantic==2.9.2

# Exponer puertos
EXPOSE 8000 8501

# Comando por defecto: levantar FastAPI y Streamlit en paralelo
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.headless true --server.port 8501


