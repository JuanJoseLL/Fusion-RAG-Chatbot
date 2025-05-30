from langchain_chroma import Chroma
from embedder import VertexAIEmbeddings
import os

def get_chroma_retriever(collection_name="rag_embeddings", k: int = 5):
    """Get ChromaDB retriever with consistent configuration.
    Args:
        collection_name (str): Name of the Chroma collection.
        k (int): Number of documents to retrieve.
    """
    # Use absolute path to ensure consistency regardless of execution directory
    persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    
    embedding_function = VertexAIEmbeddings()

    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    return vectordb.as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    """Formats retrieved documents into a single string for the LLM context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
