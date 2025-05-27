from langchain.retrievers.multi_query import MultiQueryRetriever
from config import get_logger

logger = get_logger(__name__)

class LoggedMultiQueryRetriever(MultiQueryRetriever):
    def generate_queries(self, question: str):
        queries = super().generate_queries(question)
        logger.info(f"Fusion RAG generated queries for '{question}': {queries}")
        return queries

def get_multi_query_retriever(chroma_store, llm, multi_query_k: int):
    """
    Initializes and returns a LoggedMultiQueryRetriever.
    The 'k' parameter for the underlying Chroma retriever is set by multi_query_k.
    """
    logger.info(f"Initializing LoggedMultiQueryRetriever with k={multi_query_k} for its base retriever.")
    retriever = LoggedMultiQueryRetriever.from_llm(
        retriever=chroma_store.as_retriever(search_kwargs={"k": multi_query_k}),
        llm=llm
        # include_original=True # Consider if original query results should also be included by MultiQueryRetriever itself
    )
    return retriever
