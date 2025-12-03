import pandas as pd
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

class TabularRAG:
    def __init__(self, csv_path, persist_dir="rag_store"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.emb = OllamaEmbeddings(model="nomic-embed-text")  # rápido y bueno

    def load_csv(self):
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def build_store(self, chunk_size=20):
        """Convierte el CSV en texto chunked y genera embeddings."""
        docs = []
        metadatas = []

        for i in range(0, len(self.df), chunk_size):
            chunk_df = self.df.iloc[i:i+chunk_size]
            doc_text = chunk_df.to_json(orient="records")
            docs.append(doc_text)
            metadatas.append({"row_start": i, "row_end": i+len(chunk_df)})

        # Crear vector store
        self.store = Chroma.from_texts(
            texts=docs,
            embedding=self.emb,
            metadatas=metadatas,
            persist_directory=self.persist_dir
        )

    def similarity_search(self, query, k=4):
        """Devuelve los chunks más relevantes."""
        return self.store.similarity_search(query, k=k)