#!/usr/bin/env python3
"""
Diagnostic script to inspect all ChromaDB databases and help consolidate them.
"""

import os
import sys
import chromadb

def inspect_database(db_path, db_name):
    """Inspect a ChromaDB database and report its contents."""
    print(f"\n🔍 Inspecting {db_name}: {db_path}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"❌ Database path does not exist: {db_path}")
        return {}
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # In ChromaDB v0.6.0+, we need to try known collection names
        # The error messages tell us the collection names that exist
        known_collections = ["documents", "rag_embeddings"]
        
        collection_info = {}
        found_collections = 0
        
        for collection_name in known_collections:
            try:
                collection = client.get_collection(collection_name)
                found_collections += 1
                count = collection.count()
                print(f"\n📋 Collection: {collection_name}")
                print(f"   Documents: {count}")
                
                if count > 0:
                    # Get a sample of documents
                    sample_size = min(count, 3)
                    results = collection.get(
                        limit=sample_size,
                        include=['documents', 'metadatas', 'ids']
                    )
                    
                    print(f"   Sample IDs: {results['ids'][:3]}")
                    if results['metadatas']:
                        sources = set()
                        for metadata in results['metadatas']:
                            if metadata and 'source' in metadata:
                                sources.add(metadata['source'])
                        print(f"   Sample sources: {list(sources)[:3]}")
                
                collection_info[collection_name] = {
                    'count': count,
                    'collection': collection
                }
            except Exception:
                # Collection doesn't exist, which is normal
                continue
        
        print(f"\n📊 Found {found_collections} collection(s)")
        return collection_info
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return {}

def main():
    """Main diagnostic function."""
    print("🚀 ChromaDB Diagnostic Tool")
    print("This will help you understand what data exists in each ChromaDB location.")
    
    # Database locations to check
    databases = {
        "Root ChromaDB": "./chroma_db",
        "Data Context ChromaDB": "./data/context_files/chroma_db"
    }
    
    all_db_info = {}
    
    for db_name, db_path in databases.items():
        all_db_info[db_name] = inspect_database(db_path, db_name)
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    
    total_collections = 0
    total_documents = 0
    
    for db_name, db_info in all_db_info.items():
        if db_info:
            collections_count = len(db_info)
            docs_count = sum(info['count'] for info in db_info.values())
            total_collections += collections_count
            total_documents += docs_count
            
            print(f"{db_name}:")
            print(f"  Collections: {collections_count}")
            print(f"  Total documents: {docs_count}")
            
            for collection_name, info in db_info.items():
                print(f"    - {collection_name}: {info['count']} docs")
        else:
            print(f"{db_name}: No collections found")
    
    print(f"\nTotal across all databases:")
    print(f"  Collections: {total_collections}")
    print(f"  Documents: {total_documents}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if total_documents > 0:
        print("✅ You have data in your ChromaDB databases!")
        print("🔧 After running this diagnostic:")
        print("   1. The embedder.py and retriever.py have been updated to use consistent paths")
        print("   2. They now both point to the root ./chroma_db directory")
        print("   3. They use the 'rag_embeddings' collection name")
        print("   4. If you have important data in data/context_files/chroma_db,")
        print("      you may want to migrate it to the root chroma_db")
    else:
        print("⚠️  No data found in any ChromaDB databases.")
        print("   You may need to re-ingest your documents.")
    
    print(f"\n🎯 NEXT STEPS")
    print("=" * 60)
    print("1. Run your embedding/ingestion process")
    print("2. Verify data is being stored in ./chroma_db")
    print("3. Test retrieval with the updated retriever.py")

if __name__ == "__main__":
    main() 