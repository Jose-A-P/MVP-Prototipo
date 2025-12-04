import requests
import os

class StableOllamaEmbeddings:
    def __init__(self, model="nomic-embed-text", host=None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        # Dimensión típica de nomic-embed-text (ajústala si sabes el valor exacto)
        self.vector_size = 768  

    def _safe_embed(self, text: str):
        payload = {"model": self.model, "input": text}

        for _ in range(2):  # 2 intentos
            try:
                r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=15)
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding") or data.get("embeddings") or data.get("data", [None])[0]
                if isinstance(emb, list):
                    return emb
            except Exception as e:
                print("[RAG WARNING] Error al obtener embedding, reintentando:", e)
                continue

        # fallback: vector cero del tamaño esperado
        print("[RAG WARNING] Usando embedding cero por fallo repetido.")
        return [0.0] * self.vector_size

    def embed_documents(self, texts):
        embeddings = []
        for i, t in enumerate(texts):
            if not isinstance(t, str) or t.strip() == "":
                print(f"[RAG WARNING] Texto vacío o no string en índice {i}, usando vector cero.")
                embeddings.append([0.0] * self.vector_size)
            else:
                emb = self._safe_embed(t)
                if not isinstance(emb, list):
                    print(f"[RAG WARNING] Embedding inválido en índice {i}, usando vector cero.")
                    emb = [0.0] * self.vector_size
                embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return [0.0] * self.vector_size
        return self._safe_embed(text)
