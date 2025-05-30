import os 
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import logging


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_backend.log")

os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(console_format)
    root_logger.addHandler(file_handler)
    
    return root_logger

logger = setup_logging()

def get_logger(name=None):
    """
    Get a logger with the given name.
    If no name is provided, returns the root logger.
    
    Usage in other files:
    from config import get_logger
    logger = get_logger(__name__)  # Will use the module name
    """
    if name:
        return logging.getLogger(name)
    return logger


UPLOAD_DIR = 'data/context_files'
SOURCE_FILES_DIR = "txt-files"# Directory for source files to be processed
EMBBEDING_TIMEOUT = 120  # Increased timeout for Vertex AI
EMBBEDING_RETRIES = 5   # More retries for quota handling
EMBEDDING_BATCH_SIZE = 5  # Process embeddings in smaller batches

EMBEDDING_DIMENSION = 768 # text-embedding-005 (Google Vertex AI)
#EMBEDDING_DIMENSION = 512 # text-embedding-3-large
#EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

QWEN_API_URL = os.getenv("QWEN_API_URL", "https://api.totalgpt.ai/v1/chat/completions")
API_KEY = os.getenv("INFERMATIC_API_KEY") # Ensure this is set in your .env
EMBEDDING_MODEL = "gemini-embedding-001" # Google Vertex AI embedding model
MODEL_NAME = os.getenv("MODEL_NAME", "Sao10K-72B-Qwen2.5-Kunou-v1-FP8-Dynamic") # Added default value

# --- Ollama LLM Configuration ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b") # Or your preferred Ollama model

# --- General LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") # Can be "qwen" or "ollama"

# Google Vertex AI Configuration
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")

LLM_TIMEOUT = 120 # seconds
LLM_RETRIES = 3
LLM_MAX_TOKENS = 7000
LLM_TEMPERATURE = 0.7
LLM_TOP_K = 40
LLM_REPETITION_PENALTY = 1.2

# --- RAG Config ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
GRAPH_CONTEXT_NEIGHBORS = 5 # How many NEXT neighbors to fetch (0 = none) - Likely intended for knowledge graph traversal, specifying how many neighbor nodes/documents to retrieve to expand context.
SIMPLE_RETRIEVER_K = 5 # Number of documents for the simple retriever

INGEST_SIMILARITY_THRESHOLD = 0.80 # Probably for identifying sufficiently similar documents/chunks during ingestion, e.g., to link them or flag for review.
INGEST_SIMILAR_NEIGHBORS_TO_LINK = 5 # If similarity is found during ingestion, this might define how many of the most similar items to link.
INGEST_ENABLE_INTRA_DOC_SIMILARITY = "true" # If "true", suggests that similarity checks should also be performed between chunks originating from the same document during ingestion.


# HF_MODEL_NAME =  "dslim/bert-base-NER" # No longer used
# ENTITY_LABELS_TO_EXTRACT = ["PER", "ORG", "LOC"] # No longer used

os.makedirs(UPLOAD_DIR, exist_ok=True)
