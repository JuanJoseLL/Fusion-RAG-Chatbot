#!/usr/bin/env python3
"""
Script para inspeccionar el contenido de ChromaDB y verificar qué documentos están almacenados.
"""

import os
import sys
sys.path.append('/home/juan/uni/semestre8/ia/Fusion-RAG-Chatbot/rag-backend')

from langchain_community.vectorstores import Chroma
from embedder import VertexAIEmbeddings
from config import UPLOAD_DIR

def inspect_chroma_db():
    """Inspecciona el contenido actual de ChromaDB"""
    
    print("🔍 Inspeccionando ChromaDB...")
    
    # Inicializar embeddings y Chroma
    embedding_model = VertexAIEmbeddings()
    chroma_store = Chroma(
        collection_name="rag_embeddings",
        embedding_function=embedding_model,
        persist_directory=os.path.join(UPLOAD_DIR, "chroma_db")
    )
    
    # Obtener la colección
    collection = chroma_store._collection
    
    # Información básica
    total_docs = collection.count()
    print(f"📊 Total de documentos en ChromaDB: {total_docs}")
    
    if total_docs == 0:
        print("❌ No hay documentos en la base de datos!")
        return
    
    # Obtener todos los documentos (limitado a 100 para evitar sobrecarga)
    limit = min(total_docs, 100)
    results = collection.get(
        limit=limit,
        include=['documents', 'metadatas', 'ids']
    )
    
    print(f"\n📋 Mostrando los primeros {len(results['ids'])} documentos:")
    print("=" * 80)
    
    for i, (doc_id, metadata, document) in enumerate(zip(
        results['ids'], 
        results['metadatas'], 
        results['documents']
    )):
        print(f"\n📄 Documento {i+1}:")
        print(f"   ID: {doc_id}")
        print(f"   Metadata: {metadata}")
        print(f"   Contenido (primeros 200 chars): {document[:200]}...")
        print("-" * 40)
    
    # Estadísticas por fuente
    sources = {}
    for metadata in results['metadatas']:
        source = metadata.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\n📈 Estadísticas por fuente:")
    for source, count in sources.items():
        print(f"   {source}: {count} chunks")

if __name__ == "__main__":
    inspect_chroma_db()
