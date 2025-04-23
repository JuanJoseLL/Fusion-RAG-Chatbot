from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
import os
from PyPDF2 import PdfReader
from config import get_logger, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def generate_chunk_id(text: str, index: int, source: str) -> str:
    """Genera un ID único por chunk."""
    base = f"{source}-{index}-{hashlib.md5(text.encode()).hexdigest()[:8]}"
    return base

def load_and_split_document(file_path: str) -> List[Document]:
    ext = file_path.lower()
    documents = []

    # --- Cargar documento ---
    if ext.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            documents = [Document(page_content=full_text, metadata={"source": os.path.basename(file_path)})]
            logger.info(f"Loaded PDF: {file_path} with {len(reader.pages)} pages.")
        except Exception as e:
            logger.error(f"Failed to read PDF with PyPDF2: {e}")
            return []

    elif ext.endswith(".txt"):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            logger.info(f"Loaded TXT file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load text file: {e}")
            return []
    else:
        logger.warning(f"Unsupported file type: {file_path}. Skipping.")
        return []

    if not documents:
        logger.warning(f"No content loaded from: {file_path}")
        return []

    # --- Split documents ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    # --- Agregar metadata IDs únicos ---
    for i, doc in enumerate(split_docs):
        doc.metadata["id"] = generate_chunk_id(doc.page_content, i, doc.metadata.get("source", "unknown"))
    
    return split_docs
