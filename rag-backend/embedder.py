import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from config import get_logger, EMBEDDING_MODEL as CFG_EMBEDDING_MODEL, GOOGLE_PROJECT_ID, GOOGLE_LOCATION
import chromadb
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

logger = get_logger(__name__)
load_dotenv()

# Initialize Vertex AI

aiplatform.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)

EMBEDDING_RETRIES = 5
EMBEDDING_TIMEOUT = 120  # seconds
BATCH_SIZE = 1  # Vertex AI requires batch size of 1 for text embeddings
MIN_DELAY_BETWEEN_REQUESTS = 3  # Minimum seconds between individual requests

# --- Clase del embedding para Vertex AI ---
class VertexAIEmbeddings(Embeddings):
    """Wrapper for Google Vertex AI Text Embedding API."""

    def __init__(
        self,
        model_name: str = CFG_EMBEDDING_MODEL,
        project_id: str = GOOGLE_PROJECT_ID,
        location: str = GOOGLE_LOCATION,
        retries: int = EMBEDDING_RETRIES,
        timeout: int = EMBEDDING_TIMEOUT
    ):
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.retries = retries
        self.timeout = timeout
        
        # Initialize the embedding model
        try:
            self.model = TextEmbeddingModel.from_pretrained(self.model_name)
            logger.info(f"🔗 Using Vertex AI embedding model: {self.model_name}")
            logger.info(f"🌍 Project: {self.project_id}, Location: {self.location}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI embedding model: {e}")
            raise ValueError(f"Failed to initialize Vertex AI embedding model: {e}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Vertex AI with quota management."""
        if not texts:
            logger.warning("⚠️ Attempted to get embeddings for empty text list.")
            return []

        # Process in batches to avoid quota exhaustion
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add delay between batches to respect rate limits
            if i + BATCH_SIZE < len(texts):
                time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
                logger.debug(f"⏱️ Waiting {MIN_DELAY_BETWEEN_REQUESTS}s between batches to respect quotas")
        
        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a single batch of texts with exponential backoff."""
        for attempt in range(self.retries):
            try:
                logger.info(f"🔁 Vertex AI embedding batch request attempt {attempt + 1} of {self.retries}")
                logger.debug(f"📦 Processing batch of {len(texts)} texts for embedding")
                
                # Get embeddings from Vertex AI
                embeddings_response = self.model.get_embeddings(texts)
                
                # Extract the embedding vectors
                embeddings = [embedding.values for embedding in embeddings_response]
                
                logger.info(f"✅ Successfully received {len(embeddings)} embeddings from Vertex AI.")
                return embeddings

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error during Vertex AI embedding (attempt {attempt+1}): {error_msg}")
                
                # Check if it's a quota error
                if "Quota exceeded" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    # Exponential backoff for quota errors
                    delay = min(60, (2 ** attempt) * 5)  # Cap at 60 seconds
                    logger.warning(f"🚫 Quota exceeded. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                elif "ResourceExhausted" in error_msg:
                    # Different handling for resource exhausted
                    delay = min(120, (2 ** attempt) * 10)  # Longer delay, cap at 2 minutes
                    logger.warning(f"⚠️ Resource exhausted. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    # Standard backoff for other errors
                    delay = (attempt + 1) * 2
                    time.sleep(delay)
                
                if attempt + 1 == self.retries:
                    if "Quota exceeded" in error_msg:
                        raise ValueError(f"Quota exceeded after {self.retries} attempts. Please check your Google Cloud quotas and limits.")
                    else:
                        raise ValueError(f"Failed to get embeddings after {self.retries} attempts: {e}")

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._embed([text])
        if not result:
            raise ValueError("Failed to embed query text.")
        return result[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Make the class callable for compatibility."""
        return self.embed_documents(input)

# --- Chroma setup + save util ---
def get_chroma_collection():
    """Lazy initialization of ChromaDB collection to avoid import-time errors."""
    # Use absolute path to ensure consistency regardless of execution directory
    import os
    chroma_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    
    client = chromadb.PersistentClient(path=chroma_db_path)
    embedding_function = VertexAIEmbeddings()
    
    collection = client.get_or_create_collection(
        name="rag_embeddings",  # Use consistent collection name
        embedding_function=embedding_function
    )
    return collection

def save_document(doc_id: str, content: str, metadata: dict = None):
    """Save a document to ChromaDB with Vertex AI embeddings."""
    collection = get_chroma_collection()
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata or {}]
    )
    print(f"✅ Document stored: {doc_id}")
