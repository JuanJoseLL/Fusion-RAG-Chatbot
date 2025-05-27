from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from config import get_logger, RRF_K_CONSTANT, RRF_STANDARD_K, RRF_MULTI_QUERY_K
from fusion_retriever import LoggedMultiQueryRetriever # Assuming LoggedMultiQueryRetriever is in fusion_retriever.py

logger = get_logger(__name__)

def reciprocal_rank_fusion(
    list_of_ranked_document_lists: List[List[Document]], 
    k: int = RRF_K_CONSTANT
) -> List[Document]:
    """
    Performs Reciprocal Rank Fusion on a list of ranked document lists.

    Args:
        list_of_ranked_document_lists: A list of lists, where each inner list 
                                       contains Document objects sorted by relevance.
        k: The constant used in the RRF scoring formula (default from config).

    Returns:
        A single list of Document objects, sorted by their final RRF score
        in descending order, with duplicates handled.
    """
    if not list_of_ranked_document_lists:
        return []

    ranked_results: Dict[str, float] = {}
    doc_objects: Dict[str, Document] = {}

    for doc_list in list_of_ranked_document_lists:
        for rank, doc in enumerate(doc_list):
            if not hasattr(doc, 'metadata') or 'id' not in doc.metadata:
                logger.warning(f"Document missing metadata or 'id'. Skipping. Content: {doc.page_content[:100]}")
                continue
            
            doc_id = doc.metadata['id']
            score = 1.0 / (rank + k)

            if doc_id not in ranked_results:
                ranked_results[doc_id] = 0.0
                doc_objects[doc_id] = doc  # Store the document object the first time we see it
            
            ranked_results[doc_id] += score

    # Sort documents by their aggregated RRF score in descending order
    sorted_doc_ids = sorted(ranked_results.keys(), key=lambda id: ranked_results[id], reverse=True)
    
    final_documents = [doc_objects[doc_id] for doc_id in sorted_doc_ids]
    
    logger.debug(f"RRF input lists count: {len(list_of_ranked_document_lists)}")
    logger.debug(f"RRF initial unique docs: {len(doc_objects)}")
    logger.debug(f"RRF final sorted docs count: {len(final_documents)}")
    
    return final_documents


class RRFRetriever(BaseRetriever):
    """
    A retriever that combines results from a standard vector store retriever
    and a MultiQueryRetriever using Reciprocal Rank Fusion (RRF).
    """
    chroma_store: Chroma
    llm_for_multi_query: BaseChatModel
    base_retriever_k: int
    multi_query_retriever_k: int
    rrf_k_constant: int

    standard_retriever: BaseRetriever
    multi_query_retriever: BaseRetriever

    def __init__(
        self, 
        chroma_store: Chroma, 
        llm_for_multi_query: BaseChatModel,
        base_retriever_k: int = RRF_STANDARD_K,
        multi_query_retriever_k: int = RRF_MULTI_QUERY_K, # k for each query in multi-query
        rrf_k_constant: int = RRF_K_CONSTANT
    ):
        super().__init__() # BaseRetriever does not take arguments for __init__
        self.chroma_store = chroma_store
        self.llm_for_multi_query = llm_for_multi_query
        self.base_retriever_k = base_retriever_k
        self.multi_query_retriever_k = multi_query_retriever_k
        self.rrf_k_constant = rrf_k_constant

        # Initialize internal retrievers
        self.standard_retriever = self.chroma_store.as_retriever(
            search_kwargs={"k": self.base_retriever_k}
        )
        
        # Assuming LoggedMultiQueryRetriever.from_llm can take a base retriever
        # or directly the chroma_store and search_kwargs for its internal retriever
        multi_query_base_retriever = self.chroma_store.as_retriever(
            search_kwargs={"k": self.multi_query_retriever_k}
        )
        self.multi_query_retriever = LoggedMultiQueryRetriever.from_llm(
            retriever=multi_query_base_retriever, 
            llm=self.llm_for_multi_query,
            # include_original=True # Consider if original query results should also be included by MultiQueryRetriever itself
        )
        logger.info(f"RRFRetriever initialized with standard_k={self.base_retriever_k}, multi_query_k (per query)={self.multi_query_retriever_k}")


    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Gets relevant documents by invoking standard and multi-query retrievers,
        then applying Reciprocal Rank Fusion to combine the results.
        """
        logger.info(f"RRFRetriever: Getting documents for query: '{query}'")

        # 1. Invoke standard retriever
        standard_docs = self.standard_retriever.get_relevant_documents(query)
        logger.info(f"RRFRetriever: Standard retriever returned {len(standard_docs)} documents.")

        # 2. Invoke multi-query retriever
        # MultiQueryRetriever itself might generate multiple queries and aggregate results.
        # The k-value passed to its underlying retriever is multi_query_retriever_k.
        multi_query_docs = self.multi_query_retriever.get_relevant_documents(query)
        logger.info(f"RRFRetriever: Multi-query retriever returned {len(multi_query_docs)} documents (potentially from multiple generated queries).")

        # 3. Pass these lists to the reciprocal_rank_fusion function
        all_ranked_lists = []
        if standard_docs:
            all_ranked_lists.append(standard_docs)
        if multi_query_docs:
            all_ranked_lists.append(multi_query_docs)
        
        if not all_ranked_lists:
            logger.warning("RRFRetriever: Both retrievers returned empty lists. No documents to fuse.")
            return []

        fused_documents = reciprocal_rank_fusion(
            list_of_ranked_document_lists=all_ranked_lists,
            k=self.rrf_k_constant
        )
        logger.info(f"RRFRetriever: After fusion, {len(fused_documents)} documents remaining.")
        
        # The RRF function already sorts, so no need to sort again here.
        # It also handles deduplication by 'id'.
        return fused_documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Asynchronous version, can be implemented similarly if needed
        # For now, falling back to synchronous version
        logger.info(f"RRFRetriever: Async getting documents for query: '{query}'")
        standard_docs = await self.standard_retriever.aget_relevant_documents(query)
        logger.info(f"RRFRetriever (async): Standard retriever returned {len(standard_docs)} documents.")
        
        multi_query_docs = await self.multi_query_retriever.aget_relevant_documents(query)
        logger.info(f"RRFRetriever (async): Multi-query retriever returned {len(multi_query_docs)} documents.")

        all_ranked_lists = []
        if standard_docs:
            all_ranked_lists.append(standard_docs)
        if multi_query_docs:
            all_ranked_lists.append(multi_query_docs)

        if not all_ranked_lists:
            logger.warning("RRFRetriever (async): Both retrievers returned empty lists.")
            return []
            
        # reciprocal_rank_fusion is synchronous, no await here unless it's also made async
        fused_documents = reciprocal_rank_fusion(
            list_of_ranked_document_lists=all_ranked_lists,
            k=self.rrf_k_constant
        )
        logger.info(f"RRFRetriever (async): After fusion, {len(fused_documents)} documents remaining.")
        return fused_documents

# Example usage (for testing purposes, not part of the final app logic here)
if __name__ == '__main__':
    # This block would require setting up mock objects or actual instances
    # of Chroma, LLM, etc. which is complex for a simple file creation.
    # For now, this serves as a structural placeholder.
    
    # Mock documents
    doc1 = Document(page_content="Doc A content", metadata={"id": "doc_A"})
    doc2 = Document(page_content="Doc B content", metadata={"id": "doc_B"})
    doc3 = Document(page_content="Doc C content", metadata={"id": "doc_C"})
    doc4 = Document(page_content="Doc D content", metadata={"id": "doc_D"})

    list1 = [doc1, doc2, doc3] # Retriever 1 results
    list2 = [doc2, doc1, doc4] # Retriever 2 results

    fused = reciprocal_rank_fusion([list1, list2], k=2)
    print("Fused results:")
    for doc in fused:
        print(f"ID: {doc.metadata['id']}, Content: '{doc.page_content}'")

    # Expected order (approx, depends on k and exact scores): doc1, doc2, doc4, doc3 or doc1, doc2, doc3, doc4
    # Scores:
    # doc1: (1/(0+2)) + (1/(1+2)) = 0.5 + 0.333 = 0.833
    # doc2: (1/(1+2)) + (1/(0+2)) = 0.333 + 0.5 = 0.833
    # doc3: (1/(2+2)) = 0.25
    # doc4: (1/(2+2)) = 0.25
    # If scores are identical, original order from first appearance might be preserved by sort stability,
    # or by how items are inserted into the dict then retrieved.
    # With k=60 (default for RRF_K_CONSTANT)
    # doc1: (1/60) + (1/61) = 0.01666 + 0.01639 = 0.03305
    # doc2: (1/61) + (1/60) = 0.01639 + 0.01666 = 0.03305
    # doc3: (1/62) = 0.01612
    # doc4: (1/62) = 0.01612
    # So doc1 and doc2 would be tied, then doc3 and doc4 tied.
    # The exact order for ties depends on Python's sort stability or dict iteration order.

    print("\nWith k=60 (default RRF_K_CONSTANT):")
    fused_k60 = reciprocal_rank_fusion([list1, list2]) # Uses RRF_K_CONSTANT
    for doc in fused_k60:
        print(f"ID: {doc.metadata['id']}, Content: '{doc.page_content}'")

    # Expected: doc1 and doc2 will have higher summed scores than doc3 and doc4.
    # The order between doc1 and doc2 (and doc3 and doc4) might vary if scores are identical.

    # Example of how RRFRetriever might be instantiated (conceptual)
    # from langchain_community.vectorstores import Chroma
    # from langchain_openai import ChatOpenAI # Or your CustomChatQwen
    # from embedder import InfermaticEmbeddings
    #
    # # Assume chroma_store and chat_model are initialized
    # # MOCK INITIALIZATION - Replace with actual setup
    # class MockChroma:
    #     def as_retriever(self, search_kwargs):
    #         class MockStdRetriever(BaseRetriever):
    #             def _get_relevant_documents(self, query): return [doc1,doc2]
    #             async def _aget_relevant_documents(self, query): return [doc1,doc2]
    #         return MockStdRetriever()
    #
    # class MockLLM(BaseChatModel): # Needs more methods for BaseChatModel
    #     def _generate(self, messages, stop=None, run_manager=None, **kwargs): pass
    #     def _llm_type(self) -> str: return "mock"

    # try:
    #     # rrf_retriever = RRFRetriever(
    #     #     chroma_store=MockChroma(), 
    #     #     llm_for_multi_query=MockLLM(),
    #     #     base_retriever_k=2,
    #     #     multi_query_retriever_k=1 
    #     # )
    #     # results = rrf_retriever.get_relevant_documents("test query")
    #     # print("\nRRFRetriever results:")
    #     # for doc in results:
    #     #     print(f"ID: {doc.metadata['id']}")
    #     pass
    # except Exception as e:
    #     print(f"Error in RRFRetriever example: {e}")
    #     # This will likely error if LoggedMultiQueryRetriever has complex init
    #     # or if from_llm expects a more complete LLM object.
    pass
