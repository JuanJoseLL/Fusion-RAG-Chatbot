import time
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from contextlib import asynccontextmanager # Added for lifespan manager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from embedder import InfermaticEmbeddings # save_document is not used in main.py directly
# from fusion_retriever import get_fusion_retriever # Replaced by RRFRetriever
from rrf_retriever import RRFRetriever # Import the new RRFRetriever
from document_processing import load_and_split_document
# from embedder import save_document # Duplicate import
from retriever import format_docs # get_chroma_retriever is not used in main.py directly
from model_client import CustomChatQwen
from document_processing import load_and_split_document
from langchain.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification 

from config import (
    get_logger, 
    UPLOAD_DIR, 
    # TOP_K_INITIAL_SEARCH, # Replaced by RRF specific K values
    HF_MODEL_NAME,
    MODEL_NAME,
    ENTITY_LABELS_TO_EXTRACT,
    RRF_STANDARD_K,      # New config for RRF
    RRF_MULTI_QUERY_K,   # New config for RRF
    SOURCE_FILES_DIR     # New config for source files directory
    )


logger = get_logger(__name__)


try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple" 
    )
    embedding_model = InfermaticEmbeddings() 
    chat_model = CustomChatQwen()

    chroma_store = Chroma(
        collection_name="rag_embeddings",
        embedding_function=embedding_model,
        persist_directory=os.path.join(UPLOAD_DIR, "chroma_db")
    )
    logger.info("Initialized LangChain components (Embeddings, ChatModel, Vectorstore (chroma))")
except Exception as e:
    logger.exception(f"Fatal error during initialization: {e}")
    raise RuntimeError(f"Failed to initialize core components: {e}")


def bulk_ingest_from_source_directory():
    logger.info(f"Starting bulk ingestion from directory: {SOURCE_FILES_DIR}")
    processed_files = 0
    failed_files = 0

    if not os.path.exists(SOURCE_FILES_DIR):
        logger.warning(f"Source files directory not found: {SOURCE_FILES_DIR}. Skipping bulk ingestion.")
        return

    for filename in os.listdir(SOURCE_FILES_DIR):
        file_path = os.path.join(SOURCE_FILES_DIR, filename)
        
        if not os.path.isfile(file_path):
            logger.debug(f"Skipping non-file item: {filename}")
            continue

        if filename.lower().endswith((".txt", ".pdf")):
            logger.info(f"Found supported file for ingestion: {filename}")
            try:
                # Generate document_id similar to upload_context
                # Note: ingest_pipeline itself doesn't create/save the file, 
                # it expects a path to an existing file.
                # The original upload_context saves the file first, then calls ingest_pipeline.
                # Here, the files are already in SOURCE_FILES_DIR.
                sanitized_filename = os.path.basename(filename) # Should be same as filename here
                document_id = "doc_" + sanitized_filename.replace(".", "_").replace(" ", "_")
                
                # Call ingest_pipeline directly.
                # ingest_pipeline takes file_path and document_id
                # It handles load_and_split_document and adding to ChromaDB.
                # It runs in the foreground here, not as a background task.
                ingest_pipeline(file_path, document_id)
                logger.info(f"Successfully initiated ingestion for: {filename} with doc_id: {document_id}")
                processed_files += 1
            except Exception as e:
                logger.error(f"Failed to ingest file {filename}: {e}", exc_info=True)
                failed_files += 1
        else:
            logger.debug(f"Skipping unsupported file type: {filename}")
    
    logger.info(f"Bulk ingestion complete. Processed files: {processed_files}, Failed files: {failed_files}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application startup: Initiating bulk ingestion...")
    try:
        # The bulk_ingest_from_source_directory is synchronous.
        # If it were async, we would 'await' it.
        # For now, running it directly is fine, but be aware it will block startup until complete.
        # Consider running in a separate thread if it becomes too slow for startup.
        bulk_ingest_from_source_directory()
        logger.info("Bulk ingestion process completed during startup.")
    except Exception as e:
        logger.error(f"Error during startup bulk ingestion: {e}", exc_info=True)
    
    yield
    
    # Code to run on shutdown (if any)
    logger.info("Application shutdown.")

app = FastAPI(title="Fusion-RAG Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_entities(text: str) -> Dict[str, Dict[str, Any]]:
    """Extracts named entities using Hugging Face transformers pipeline."""
    entities = {}
    allowed_labels = set(ENTITY_LABELS_TO_EXTRACT) if ENTITY_LABELS_TO_EXTRACT else None

    try:
        
        ner_results = ner_pipeline(text)
        
        for ent in ner_results:
            entity_label = ent['entity_group']
            # The 'word' field contains the extracted entity text after aggregation
            entity_text = ent['word'].strip()

            # Skip if label is not in our allowed list (if list exists)
            if allowed_labels and entity_label not in allowed_labels:
                continue

            
            if not entity_text or len(entity_text) < 2: 
                 continue

            
            entity_key = entity_text.lower()

            # Store the entity if it's new, or potentially update if score is higher
            if entity_key not in entities or float(ent['score']) > entities[entity_key].get('score', 0.0):
                 entities[entity_key] = {
                     "label": entity_label,
                     "text": entity_text, 
                     "score": float(ent['score']) 
                 }

    except Exception as e:
        
        logger.error(f"Error during Hugging Face NER processing for text segment: {e}")
        
        return {} 

    return entities

def ingest_pipeline(file_path: str, document_id: str):
    """ingestion pipeline with entity extraction (HF) and optimized similarity linking."""
    start_time = time.time()
    try:
        logger.info(f"[Ingest:{document_id}] Starting for: {file_path}")
        split_docs = load_and_split_document(file_path)
        if not split_docs:
            logger.error(f"[Ingest:{document_id}] No documents generated. Aborting.")
            return

        chunk_count = len(split_docs)
        logger.info(f"[Ingest:{document_id}] Split into {chunk_count} chunks.")

        # --- Step 1: Add Chunks to Vector Store ---
        chunk_ids = [d.metadata["id"] for d in split_docs]
        added_ids = chroma_store.add_documents(split_docs, ids=chunk_ids)
        logger.info(f"[Ingest:{document_id}] Added {len(added_ids)} chunks to Chroma vector store.")
        if not added_ids:
             logger.warning(f"[Ingest:{document_id}] No chunks added. Aborting.")
             return

        docs_to_process = [doc for doc in split_docs if doc.metadata["id"] in added_ids]

        # --- Completion ---
        end_time = time.time()
        logger.info(f"[Ingest:{document_id}] Successfully completed ingestion for: {file_path} in {end_time - start_time:.2f} seconds")

    except Exception as e:
        logger.exception(f"[Ingest:{document_id}] Critical error during ingestion pipeline for {file_path}: {e}")


# --- API Endpoints ---
@app.post("/upload-context")
async def upload_context(background_tasks: BackgroundTasks, file: UploadFile):
    sanitized_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    document_id = "doc_" + sanitized_filename.replace(".", "_").replace(" ", "_")

    # Save file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {sanitized_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
         if hasattr(file, 'file') and hasattr(file.file, 'close'):
             file.file.close()

    # --- Schedule ingestion as background task ---
    background_tasks.add_task(ingest_pipeline, file_path, document_id)

    return {
        "status": "processing",
        "message": f"File '{sanitized_filename}' received and scheduled for ingestion.",
        "document_id": document_id
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
        user_message = payload.get("messages", [])[-1]["content"]

        # Instantiate RRFRetriever
        rrf_retriever_instance = RRFRetriever(
            chroma_store=chroma_store,
            llm_for_multi_query=chat_model,
            base_retriever_k=RRF_STANDARD_K,
            multi_query_retriever_k=RRF_MULTI_QUERY_K
            # RRF_K_CONSTANT is used internally by RRFRetriever
        )
        logger.info(f"Initialized RRFRetriever with standard_k={RRF_STANDARD_K}, multi_query_k={RRF_MULTI_QUERY_K}")


        template = """You are a helpful assistant that is an expert in multiple categories. 
            Including: NVIDIA data center GPUs, Machine Learning, Artificial Intelligence, High Performance
            Computing, Large Language Models, Cloud Computing, CAD, and other related technology. Provide the best answer to the question provided.
            If a question is outside your expertise, decline to answer and ask the user to ask a question 
            that falls in your expertise. DO NOT answer the question if it is outside your expertise and never mention the documents where you got the information. 

            For NVIDIA GPUs you want to promote: RTX A30, RTX A5000, A40, RTX A6000, RTX A6000 ADA, L40, 
            L40S, A100 (80GB) PCIe, H100 PCIe, H100 NVL, A100 SXM4, H100 SXM5.

            # CRITICAL INSTRUCTIONS:
            - Restate the question in a h2 section.
            - For unordered lists, use <ul> tags with <li> list items. 
            - For ordered lists, use <ol> tags with <li> list items. 
            - For bold text, use <strong> tags. 
            - For italics, use <em> tags.
            - For headings, use appropriate <h2>, <h3>, <h4> tags.
            - For paragraphs, use <p> tags after headings.
            - For tables, use <table> tags with <tr> and <td> tags.
            - Don't use asterisks in the answer.
            - Make your answer around 600 words.
            - Don't link to other websites.
            - Don't offer any downloads. 
            - NEVER use ** in your answer.

            Use SEO best practices in the answer for the domain https://massedcompute.com/. Don't list the keywords. 

            Don't include a call to action, social media links, copyright, or other typical footer links. They're already included in a different section. 
            Format the answer with HTML that will be embedded in an existing webpage using the following example.

            ## Example output:
            <h2>Question Restated</h2>
            <p>Your answer starts here...</p>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
            <p>More text with <strong>bold text</strong> and <em>italic text</em> words.</p>

            Remember: ONLY use HTML tags for formatting. If you're unsure, use simple <p> tags for paragraphs and avoid complex formatting. 

        Context:
        {context}

        Question:
        {question}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": rrf_retriever_instance | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | chat_model
            | StrOutputParser()
        )

        reply = await rag_chain.ainvoke(user_message)

        # Retrieve usage data from the chat model instance
        usage_data = chat_model.get_last_usage_data()
        
        if not usage_data: # Provide a default if no usage data was captured
            usage_data = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "detail": "Usage data not available or not provided by API."
            }
        logger.info(f"Token usage for request: {usage_data}")

        return {
            "id": "chatcmpl-mockid", # Consider generating a unique ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME, # This should ideally be self.model_name from chat_model
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply,
                },
                "finish_reason": "stop" # Or determine this from API if available
            }],
            "usage": usage_data
        }

    except Exception as e:
        logger.exception(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Chat error")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 