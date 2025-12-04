from api.rag_engine import DigitalTwinRAG

# Singleton global
_rag_instance = None


def build_rag_pipeline():
    """
    Construye el motor RAG una sola vez.
    Si ya existe, simplemente lo regresa.
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = DigitalTwinRAG(
            persist_directory="./chroma_finance",
            embedding_host="http://host.docker.internal:11434"
        )
        _rag_instance.build_store()
        _rag_instance.build_qa_chain()

    return _rag_instance


def get_rag_answer(query: str) -> str:
    """
    Llama al motor RAG y devuelve únicamente la respuesta,
    SIN mostrar el query ni metadatos.
    """
    rag = build_rag_pipeline()
    qa = rag.get_qa_chain()

    try:
        result = qa.invoke({"query": query})
    except Exception as e:
        return f"Error en RAG: {e}"

    if isinstance(result, dict):
        return result.get("result") or result.get("answer") or str(result)

    return str(result)
