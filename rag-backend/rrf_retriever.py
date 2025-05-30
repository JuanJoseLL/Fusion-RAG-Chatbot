from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
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
    base_retriever_k: int = RRF_STANDARD_K
    multi_query_retriever_k: int = RRF_MULTI_QUERY_K
    rrf_k_constant: int = RRF_K_CONSTANT
    
    # These will be computed fields, not part of the constructor
    standard_retriever: BaseRetriever = None
    multi_query_retriever: BaseRetriever = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self, 
        chroma_store: Chroma, 
        llm_for_multi_query: BaseChatModel,
        base_retriever_k: int = RRF_STANDARD_K,
        multi_query_retriever_k: int = RRF_MULTI_QUERY_K,
        rrf_k_constant: int = RRF_K_CONSTANT,
        **kwargs
    ):
        super().__init__(
            chroma_store=chroma_store,
            llm_for_multi_query=llm_for_multi_query,
            base_retriever_k=base_retriever_k,
            multi_query_retriever_k=multi_query_retriever_k,
            rrf_k_constant=rrf_k_constant,
            **kwargs
        )

        # Initialize internal retrievers after parent initialization
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