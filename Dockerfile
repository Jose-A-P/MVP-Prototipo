FROM python:3.11-slim

RUN apt-get update && apt-get install -y tini && apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade --ignore-installed -r requirements.txt

COPY . .

ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_EMBEDDING_URL=http://host.docker.internal:11434

EXPOSE 8501

ENTRYPOINT ["tini", "--"]

CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501"]
