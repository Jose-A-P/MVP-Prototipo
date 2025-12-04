from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from api.ollama_embeddings_fix import StableOllamaEmbeddings
from api.rag_finance_knowledge import build_financial_knowledge_corpus

import os

# Configuración del host para Ollama, necesaria para llamadas de inferencia y embeddings
os.environ["OLLAMA_HOST"] = "http://host.docker.internal:11434"
os.environ["OLLAMA_EMBEDDING_URL"] = "http://host.docker.internal:11434"


class DigitalTwinRAG:
    """
    Clase que encapsula todo el sistema RAG utilizado por la aplicación.
    Este motor combina extracción de información, embeddings y un modelo LLM,
    para responder preguntas conceptuales relacionadas con el portafolio simulado.
    """

    def __init__(self, persist_directory: str | None = None, embedding_host=None):
        # Ruta donde se almacenan los vectores de embeddings
        self.persist_directory = persist_directory

        # Vectorstore donde se guardan documentos y vectores
        self.vectorstore = None

        # Objeto que maneja la recuperación de documentos relevantes
        self.retriever = None

        # Cadena de preguntas y respuestas basada en recuperación
        self.qa_chain = None

        # Host donde corre el servidor Ollama
        self.embedding_host = embedding_host or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

    def build_store(self):
        """
        Construye el almacenamiento vectorial a partir del corpus financiero generado dinámicamente.
        Se encarga de limpiar textos, generar embeddings y persistirlos.
        Este proceso se ejecuta una sola vez durante la vida de la aplicación.
        """

        print("Usando OLLAMA_HOST:", os.getenv("OLLAMA_HOST"))

        # Obtiene los textos que forman el corpus financiero
        texts = build_financial_knowledge_corpus()

        # Limpieza del corpus para evitar datos vacíos o no válidos
        clean_texts = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                print(f"[RAG WARNING] Documento {i} no es string. Se ignora.")
                continue
            if t.strip() == "":
                print(f"[RAG WARNING] Documento vacío en índice {i}. Se ignora.")
                continue
            clean_texts.append(t)

        # Validación para evitar un vectorstore vacío
        if len(clean_texts) == 0:
            raise ValueError("El corpus para RAG quedó vacío después de limpiar.")

        # Se genera una metadata simple por documento
        metadatas = [{"idx": i} for i in range(len(clean_texts))]

        # Uso de embeddings personalizados con manejo estable de errores
        embeddings_fn = StableOllamaEmbeddings(model="nomic-embed-text")
        embeddings = embeddings_fn.embed_documents(clean_texts)

        print("[RAG DEBUG] #docs:", len(clean_texts), " #embeddings:", len(embeddings))

        # Creación del vectorstore, asegurando que cada texto tenga un embedding correspondiente
        self.vectorstore = Chroma.from_embeddings(
            texts=clean_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            collection_name="digital_twin_finance",
            persist_directory=self.persist_directory
        )

        # Se construye el objeto recuperador que permitirá obtener documentos relevantes
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        return self.vectorstore

    def build_qa_chain(self):
        """
        Esta cadena toma preguntas en lenguaje natural, recupera documentos relevantes,
        y los envía a un modelo LLM para generar una respuesta con contexto.
        """

        # Si no existe un vectorstore aún, se construye
        if self.retriever is None:
            self.build_store()

        # Se inicializa el modelo de lenguaje utilizado como generador final
        llm = OllamaLLM(
            model="mistral",
            base_url=self.embedding_host
        )

        # Plantilla de prompt que define cómo debe estructurarse la respuesta
        prompt = ChatPromptTemplate.from_template(
            """
            Eres un analista de riesgo de crédito en un banco.
            Usa exclusivamente el contexto proporcionado para responder.

            CONTEXTO:
            {context}

            PREGUNTA:
            {question}

            Responde de forma clara, precisa y con terminología financiera.
            """
        )

        # Se establece el pipeline de recuperación y respuesta
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        return self.qa_chain

    def get_qa_chain(self):
        """
        Devuelve la cadena de preguntas y respuestas, construyéndola si es necesario.
        Esto permite que otras partes de la aplicación obtengan respuestas del motor RAG.
        """

        if self.qa_chain is None:
            return self.build_qa_chain()

        return self.qa_chain
