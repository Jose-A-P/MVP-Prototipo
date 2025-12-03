from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from api.rag_finance_knowledge import build_financial_knowledge_corpus


class DigitalTwinRAG:
    def __init__(self, persist_directory: str | None = None):
        self.persist_directory = persist_directory
        self.store = None
        self.retriever = None
        self.qa_chain = None

    def build_store(self):
        texts = build_financial_knowledge_corpus()

        docs = [Document(page_content=t, metadata={"source": "financial_knowledge"}) for t in texts]

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="digital_twin_finance",
            persist_directory=self.persist_directory
        )
        self.retriever = self.store.as_retriever(
            search_kwargs={"k": 4}
        )

    def build_qa_chain(self):
        if self.retriever is None:
            self.build_store()

        llm = OllamaLLM(model="mistral")

        prompt = ChatPromptTemplate.from_template(
            """
            Eres un analista de riesgo de crédito en un banco.
            Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE
            el contexto proporcionado, que describe la cartera, los clusters,
            las variables del modelo de impago y los escenarios simulados.

            Sé específico, numérico cuando sea posible, y usa lenguaje financiero claro.

            CONTEXTO:
            {context}

            PREGUNTA:
            {question}
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
