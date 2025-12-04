from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from api.ollama_embeddings_fix import StableOllamaEmbeddings

import os
os.environ["OLLAMA_HOST"] = "http://host.docker.internal:11434"
os.environ["OLLAMA_EMBEDDING_URL"] = "http://host.docker.internal:11434"

from api.rag_finance_knowledge import build_financial_knowledge_corpus


class DigitalTwinRAG:
    def __init__(self, persist_directory: str | None = None, embedding_host=None):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def build_store(self):
        """
        Construye el vectorstore a partir del corpus financiero.
        Se llama SOLO UNA VEZ desde rag_chat.build_rag_pipeline()
        """
        print("Usando OLLAMA_HOST:", os.getenv("OLLAMA_HOST"))
        
        texts = build_financial_knowledge_corpus()
        
        clean_texts = []

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                print(f"[RAG WARNING] Documento {i} no es string. Se ignora.")
                continue
            if t.strip() == "":
                print(f"[RAG WARNING] Documento vacío en índice {i}. Se ignora.")
                continue
            clean_texts.append(t)

        if len(clean_texts) == 0:
            raise ValueError("El corpus para RAG quedó vacío después de limpiar.")
        
        docs = [ Document(page_content=t, metadata={"idx": i}) for i, t in enumerate(clean_texts)]

        embeddings = StableOllamaEmbeddings(model="nomic-embed-text")

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="digital_twin_finance",
            persist_directory=self.persist_directory
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        return self.vectorstore

    def build_qa_chain(self):
        """
        Construye la cadena de QA SOLO SI hace falta.
        El retriever y vectorstore ya deben estar construidos.
        """

        if self.retriever is None:
            self.build_store()

        llm = OllamaLLM(
            model="mistral",
            base_url=self.embedding_host   # mismo host
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Eres un analista de riesgo de crédito en un banco.
            Usa EXCLUSIVAMENTE el contexto proporcionado para responder.

            CONTEXTO:
            {context}

            PREGUNTA:
            {question}

            Responde de forma clara, precisa y con terminología financiera.
            """
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        return self.qa_chain

    def get_qa_chain(self):
        if self.qa_chain is None:
            return self.build_qa_chain()
        return self.qa_chain