from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from api.rag_engine import TabularRAG

def build_rag_pipeline():
    rag = TabularRAG("data/clientes_simulados_combined.csv")
    rag.load_csv()
    rag.build_store()

    retriever = rag.store.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(model="mistral")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres un analista financiero experto.

Contexto relevante:
{context}

Pregunta del usuario:
{question}

Responde basado EN EL CONTEXTO de forma breve y de manera textual, nunca presentes codigo para calcular campos. 
Si se requiere calcular un escenario o probabilidad, explica qué función llamar.
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain
