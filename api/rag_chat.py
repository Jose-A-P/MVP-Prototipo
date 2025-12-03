from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from api.rag_engine import DigitalTwinRAG

_rag_instance = None

def build_rag_pipeline():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = DigitalTwinRAG(persist_directory="./chroma_finance")
        _rag_instance.build_store()
    return _rag_instance
