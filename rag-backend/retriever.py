from langchain.vectorstores import Chroma
from embedder import InfermaticEmbeddings

def get_chroma_retriever(persist_directory="./chroma_db", collection_name="documents"):
    embedding_function = InfermaticEmbeddings()

    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    return vectordb.as_retriever()

def format_docs(docs):
    """Formats retrieved documents into a single string for the LLM context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
