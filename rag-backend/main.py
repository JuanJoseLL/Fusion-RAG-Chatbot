import time
import os
import shutil
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from contextlib import asynccontextmanager # Added for lifespan manager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from embedder import VertexAIEmbeddings # Updated to use Vertex AI embeddings
# from fusion_retriever import get_fusion_retriever # Replaced by RRFRetriever
from rrf_retriever import RRFRetriever # Import the new RRFRetriever
from document_processing import load_and_split_document
# from embedder import save_document # Duplicate import
from retriever import format_docs # get_chroma_retriever is not used in main.py directly
from model_client import CustomChatQwen
from document_processing import load_and_split_document
from langchain_community.vectorstores import Chroma
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
    embedding_model = VertexAIEmbeddings() 
    chat_model = CustomChatQwen()

    chroma_store = Chroma(
        collection_name="rag_embeddings",
        embedding_function=embedding_model,
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
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

    # Get all supported files first
    supported_files = []
    for filename in os.listdir(SOURCE_FILES_DIR):
        file_path = os.path.join(SOURCE_FILES_DIR, filename)
        
        if not os.path.isfile(file_path):
            logger.debug(f"Skipping non-file item: {filename}")
            continue

        if filename.lower().endswith((".txt", ".pdf")):
            supported_files.append((filename, file_path))
        else:
            logger.debug(f"Skipping unsupported file type: {filename}")
    
    logger.info(f"Found {len(supported_files)} supported files for ingestion")
    
    # Process files with delays to respect API quotas
    for i, (filename, file_path) in enumerate(supported_files):
        logger.info(f"Found supported file for ingestion ({i+1}/{len(supported_files)}): {filename}")
        try:
            # Generate document_id similar to upload_context
            sanitized_filename = os.path.basename(filename)
            document_id = "doc_" + sanitized_filename.replace(".", "_").replace(" ", "_")
            
            # Call ingest_pipeline directly
            ingest_pipeline(file_path, document_id)
            logger.info(f"Successfully completed ingestion for: {filename} with doc_id: {document_id}")
            processed_files += 1
            
            # Add delay between files to respect API quotas (except for the last file)
            if i < len(supported_files) - 1:
                delay = 5  # 5 seconds between files
                logger.info(f"⏱️ Waiting {delay} seconds before processing next file to respect API quotas...")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Failed to ingest file {filename}: {e}", exc_info=True)
            failed_files += 1
            
            # If we hit quota issues, wait longer before next file
            if "Quota exceeded" in str(e) or "ResourceExhausted" in str(e):
                delay = 30  # 30 seconds after quota errors
                logger.warning(f"🚫 Quota error detected. Waiting {delay} seconds before continuing...")
                time.sleep(delay)
    
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
        # bulk_ingest_from_source_directory()
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


        template = """Eres un asistente especializado en seguros y pólizas de seguros que trabaja para una compañía de seguros. 
            Tienes experiencia en: Seguros de Vida, Accidentes Personales, Seguros de Desempleo, Enfermedades Graves, 
            Muerte Accidental, Seguros Bancarios, Pólizas de Fraude, Incapacidad Total Temporal, Seguros de Vivienda, 
            Procedimientos de Reclamaciones, Vinculación de Asociados, Condiciones Técnicas de Seguros, y Manuales de Suscripción.
            
            Proporciona la mejor respuesta posible a la pregunta basándote en tu conocimiento sobre seguros y pólizas.
            Si una pregunta está fuera de tu área de expertise en seguros, declina responder cortésmente y pide al usuario 
            que haga una pregunta relacionada con seguros, pólizas, reclamaciones o procedimientos de seguros. 
            NO respondas preguntas que no estén relacionadas con seguros y NUNCA menciones los documentos de donde obtuviste la información.

            Tu objetivo es ayudar a asociados y clientes con información sobre sus pólizas, procedimientos de reclamación, 
            coberturas disponibles, requisitos de documentación, tiempos de respuesta, y cualquier otro tema relacionado con seguros.

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