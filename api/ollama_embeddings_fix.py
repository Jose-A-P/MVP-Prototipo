import requests
import os

class StableOllamaEmbeddings:
    """
    Clase diseñada para proporcionar un sistema estable y tolerante a fallos
    al solicitar embeddings al servidor de Ollama.

    Este wrapper actúa como capa de seguridad, reintentando solicitudes y 
    ofreciendo un vector de fallback en caso de error, evitando que 
    el pipeline completo de RAG colapse.
    """

    def __init__(self, model="nomic-embed-text", host=None):
        self.model = model

        # El host se obtiene del parámetro, de la variable de entorno
        # o de un valor por defecto si ninguno está configurado.
        self.host = host or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

        # Dimensión estándar del modelo nomic-embed-text.
        # Si el modelo cambia, este valor debe actualizarse.
        self.vector_size = 768

    def _safe_embed(self, text: str):
        """
        Genera un embedding de forma robusta. Si Ollama falla o se cae
        la conexión, se reintenta una vez más antes de devolver un vector cero.
        """

        payload = {"model": self.model, "input": text}

        # Se realizan dos intentos como mecanismo de resiliencia
        for _ in range(2):
            try:
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json=payload,
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()

                # Algunos servidores devuelven "embedding", otros "embeddings"
                emb = (
                    data.get("embedding") or
                    data.get("embeddings") or
                    data.get("data", [None])[0]
                )

                if isinstance(emb, list):
                    return emb

            except Exception as e:
                print("[RAG WARNING] Error al obtener embedding. Reintentando:", e)
                continue

        # Si ambos intentos fallaron, se devuelve un vector cero
        print("[RAG WARNING] Usando embedding cero por fallo repetido.")
        return [0.0] * self.vector_size

    def embed_documents(self, texts):
        """
        Genera embeddings para múltiples textos, aplicando validación
        individual y fallback seguro para cada documento.
        """

        embeddings = []

        for idx, t in enumerate(texts):
            # Validación básica para evitar fallos en el pipeline
            if not isinstance(t, str) or t.strip() == "":
                print(f"[RAG WARNING] Texto vacío o inválido en índice {idx}. Usando embedding cero.")
                embeddings.append([0.0] * self.vector_size)
                continue

            vec = self._safe_embed(t)

            if not isinstance(vec, list):
                print(f"[RAG WARNING] Embedding inválido en índice {idx}. Usando embedding cero.")
                vec = [0.0] * self.vector_size

            embeddings.append(vec)

        return embeddings

    def embed_query(self, text):
        """
        Genera un embedding para una consulta individual que el modelo
        de lenguaje utilizará para buscar en el vector store.
        """

        if not isinstance(text, str) or text.strip() == "":
            return [0.0] * self.vector_size

        return self._safe_embed(text)
