from api.rag_engine import DigitalTwinRAG

# Se declara una instancia global para asegurar que el motor RAG
# solo se construya una vez durante toda la sesión.
_rag_instance = None


def build_rag_pipeline():
    """
    Crea e inicializa el motor RAG si aún no existe.
    """

    global _rag_instance

    # Si aún no existe una instancia del motor, se crea una nueva
    if _rag_instance is None:
        _rag_instance = DigitalTwinRAG(
            persist_directory="./chroma_finance",
            embedding_host="http://host.docker.internal:11434"
        )

        # Construcción del vector store usando los documentos financieros
        _rag_instance.build_store()

        # Construcción de la cadena de preguntas y respuestas
        _rag_instance.build_qa_chain()

    return _rag_instance


def get_rag_answer(query: str) -> str:
    """
    Obtiene una respuesta desde el motor RAG, aislando al usuario de detalles internos.
    La función solo entrega el texto final producido por el modelo,
    ocultando estructuras internas.
    """

    # Se garantiza que exista un motor configurado
    rag = build_rag_pipeline()

    # Se obtiene la cadena QA ya lista para responder preguntas
    qa = rag.get_qa_chain()

    # Se intenta ejecutar la consulta; si algo falla se captura el error
    try:
        result = qa.invoke({"query": query})
    except Exception as e:
        return f"Error en RAG: {e}"

    # Las implementaciones de LangChain pueden devolver un diccionario;
    # aquí se extrae solo el texto relevante.
    if isinstance(result, dict):
        return result.get("result") or result.get("answer") or str(result)

    # Si el resultado no es un diccionario, se devuelve tal cual
    return str(result)
