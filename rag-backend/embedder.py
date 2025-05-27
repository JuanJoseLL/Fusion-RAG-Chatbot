import os
import requests
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from config import get_logger, EMBEDDING_MODEL as CFG_EMBEDDING_MODEL # Import from config
import chromadb

logger = get_logger(__name__)
load_dotenv()

INFERMATIC_EMBEDDINGS_ENDPOINT = os.getenv("INFERMATIC_EMBEDDINGS_ENDPOINT")
INFERMATIC_API_KEY = os.getenv("INFERMATIC_API_KEY")
# EMBEDDING_MODEL is now imported from config.py
EMBEDDING_RETRIES = 3
EMBEDDING_TIMEOUT = 90  # seconds

# --- Clase del embedding ---
class InfermaticEmbeddings(Embeddings):
    """Wrapper for Infermatic Embeddings API service."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = CFG_EMBEDDING_MODEL, # Use the imported config value
        retries: int = EMBEDDING_RETRIES,
        timeout: int = EMBEDDING_TIMEOUT
    ):
        self.endpoint = endpoint or INFERMATIC_EMBEDDINGS_ENDPOINT
        self.api_key = api_key or INFERMATIC_API_KEY
        self.model = model # This will now correctly use the value from config
        self.retries = retries
        self.timeout = timeout

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing Infermatic embeddings endpoint or API key. "
                "Set INFERMATIC_EMBEDDINGS_ENDPOINT and INFERMATIC_API_KEY environment variables."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"🔗 Using Infermatic API endpoint: {self.endpoint}")
        logger.info(f"🔐 API Key starts with: {self.api_key[:5]}...")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("⚠️ Attempted to get embeddings for empty text list.")
            return []

        payload = {
            "input": texts,
            "model": self.model, # Use self.model here
        }

        logger.debug(f"📦 Embedding request payload: {payload}")
        logger.debug(f"🧾 Headers: { {k: ('***' if k.lower() == 'authorization' else v) for k, v in self.headers.items()} }")

        for attempt in range(self.retries):
            try:
                logger.info(f"🔁 Embedding request attempt {attempt + 1} of {self.retries}")
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                logger.debug(f"📡 Response status code: {response.status_code}")
                logger.debug(f"📨 Response body: {response.text}")

                response.raise_for_status()  # This will raise if 4xx or 5xx

                response_data = response.json()

                if "data" not in response_data:
                    logger.error(f"❌ Response JSON missing 'data' field: {response_data}")
                    raise ValueError("Response JSON missing 'data' field.")

                sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in sorted_data]
                logger.info(f"✅ Successfully received {len(embeddings)} embeddings.")
                return embeddings

            except requests.HTTPError as e:
                logger.error(f"❌ HTTPError during Infermatic embedding (attempt {attempt+1}): {e}")
                logger.error(f"🔍 Response: {response.text}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Failed to get embeddings after {self.retries} attempts: {e}")
                time.sleep(1 * (attempt + 1))

            except Exception as e:
                logger.error(f"💥 Unexpected error (attempt {attempt+1}): {e}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Unexpected error while getting embeddings: {e}")
                time.sleep(1 * (attempt + 1))

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._embed([text])
        if not result:
            raise ValueError("Failed to embed query text.")
        return result[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

# --- Chroma setup + save util ---
client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = InfermaticEmbeddings()

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)

def save_document(doc_id: str, content: str, metadata: dict = None):
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata or {}]
    )
    print(f"✅ Document stored: {doc_id}")
