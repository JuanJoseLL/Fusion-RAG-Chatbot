#!/usr/bin/env python3
"""
Consolidation script to merge all ChromaDB databases into one single location.
This will create ONE database with consistent configuration.
"""

import os
import sys
import chromadb
import shutil
from pathlib import Path

def consolidate_databases():
    """Consolidate all ChromaDB data into one single database."""
    print("🚀 ChromaDB Consolidation Tool")
    print("This will create ONE database at ./chroma_db with collection 'rag_embeddings'")
    print("=" * 70)
    
    # Define paths
    root_db_path = "./chroma_db"
    data_db_path = "./data/context_files/chroma_db"
    
    # Target configuration (what we want to end up with)
    target_db_path = root_db_path
    target_collection_name = "rag_embeddings"
    
    print(f"🎯 Target: {target_db_path} with collection '{target_collection_name}'")
    
    # Initialize target database
    print(f"\n📦 Setting up target database...")
    target_client = chromadb.PersistentClient(path=target_db_path)
    
    # Try to get or create the target collection
    try:
        target_collection = target_client.get_collection(target_collection_name)
        print(f"✅ Found existing collection '{target_collection_name}' with {target_collection.count()} documents")
    except:
        print(f"🆕 Creating new collection '{target_collection_name}'")
        # We'll create it when we have embeddings to add
        target_collection = None
    
    # Migrate data from data context DB
    total_migrated = 0
    if os.path.exists(data_db_path):
        print(f"\n🔄 Checking data context database...")
        try:
            source_client = chromadb.PersistentClient(path=data_db_path)
            source_collection = source_client.get_collection("rag_embeddings")
            source_count = source_collection.count()
            
            if source_count > 0:
                print(f"📥 Found {source_count} documents to migrate")
                
                # Get all data from source
                source_data = source_collection.get(
                    include=['documents', 'metadatas', 'embeddings']
                )
                
                if not target_collection:
                    # Create target collection with same embedding function
                    print(f"🆕 Creating target collection...")
                    # We need to create it properly - let's do this through the embedder
                    sys.path.append('./rag-backend')
                    from embedder import VertexAIEmbeddings
                    
                    embedding_function = VertexAIEmbeddings()
                    target_collection = target_client.get_or_create_collection(
                        name=target_collection_name,
                        embedding_function=embedding_function
                    )
                
                # Add data to target collection
                print(f"📤 Migrating {len(source_data['ids'])} documents...")
                target_collection.add(
                    ids=source_data['ids'],
                    documents=source_data['documents'],
                    metadatas=source_data['metadatas'],
                    embeddings=source_data['embeddings']
                )
                
                total_migrated = len(source_data['ids'])
                print(f"✅ Successfully migrated {total_migrated} documents!")
                
            else:
                print(f"ℹ️  Source collection is empty, nothing to migrate")
                
        except Exception as e:
            print(f"⚠️  Could not access data context database: {e}")
    else:
        print(f"ℹ️  No data context database found at {data_db_path}")
    
    # Clean up old collections in root DB
    print(f"\n🧹 Cleaning up old collections...")
    try:
        old_collection = target_client.get_collection("documents")
        old_count = old_collection.count()
        if old_count == 0:
            print(f"🗑️  Removing empty 'documents' collection")
            target_client.delete_collection("documents")
        else:
            print(f"⚠️  Found {old_count} documents in 'documents' collection - keeping it for manual review")
    except:
        print(f"ℹ️  No 'documents' collection to clean up")
    
    # Final verification
    print(f"\n✅ CONSOLIDATION COMPLETE!")
    print("=" * 70)
    
    try:
        final_collection = target_client.get_collection(target_collection_name)
        final_count = final_collection.count()
    except:
        print(f"⚠️  Collection '{target_collection_name}' was not created (no data to migrate)")
        final_count = 0
    
    print(f"📊 Final database: {target_db_path}")
    print(f"📋 Collection: {target_collection_name}")
    print(f"📄 Total documents: {final_count}")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. All your code now uses the consistent configuration:")
    print(f"   - Database: {target_db_path}")
    print(f"   - Collection: {target_collection_name}")
    print("2. Test your embedding/retrieval process")
    print("3. Consider removing the old data context database if migration was successful")
    
    return final_count

if __name__ == "__main__":
    consolidate_databases() 