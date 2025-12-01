FROM python:3.11-slim

WORKDIR /app
COPY . /app


#Actualiza pip dentro del contenedor
RUN pip install --upgrade pip setuptools wheel


# Instalar dependencias de Python desde requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Variables de entorno
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV API_URL=http://localhost:8000

# Exponer puertos
EXPOSE 8000 8501

# Comando por defecto: levantar FastAPI y Streamlit en paralelo
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
           streamlit run app.py --server.headless true --server.port 8501"

