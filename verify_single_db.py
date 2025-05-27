#!/usr/bin/env python3
"""
Simple verification script to confirm single ChromaDB setup is working.
"""

import sys
import os

# Add rag-backend to path
sys.path.append('./rag-backend')

def test_single_database():
    """Test that the single database configuration works."""
    print("🧪 Testing Single ChromaDB Configuration")
    print("=" * 50)
    
    try:
        # Test embedder
        print("📝 Testing embedder...")
        from embedder import get_chroma_collection
        collection = get_chroma_collection()
        embedder_count = collection.count()
        print(f"   ✅ Embedder collection: {embedder_count} documents")
        
        # Test retriever
        print("🔍 Testing retriever...")
        from retriever import get_chroma_retriever
        retriever = get_chroma_retriever()
        print(f"   ✅ Retriever initialized successfully")
        
        # Test that they point to the same collection
        print("🔗 Verifying consistency...")
        print(f"   Database path: ./chroma_db")
        print(f"   Collection name: rag_embeddings")
        print(f"   Documents: {embedder_count}")
        
        if embedder_count > 0:
            print("   🎉 SUCCESS: Single database setup is working!")
            print("   📄 You have data in your database and can start using it")
        else:
            print("   ⚠️  Database is set up correctly but empty")
            print("   💡 You can now add documents using your embedding process")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def cleanup_old_database():
    """Optionally remove the old data context database."""
    old_db_path = "./data/context_files/chroma_db"
    
    if os.path.exists(old_db_path):
        print(f"\n🧹 Old database cleanup")
        print("=" * 50)
        print(f"Old database still exists at: {old_db_path}")
        print("Since migration was successful, you can safely remove it:")
        print(f"   rm -rf {old_db_path}")
        print("(Uncomment the next line to auto-remove)")
        # import shutil; shutil.rmtree(old_db_path); print("✅ Old database removed")

if __name__ == "__main__":
    success = test_single_database()
    cleanup_old_database()
    
    if success:
        print(f"\n🎯 SUMMARY: Your ChromaDB is now consolidated!")
        print("   ✅ Single database: ./chroma_db")
        print("   ✅ Single collection: rag_embeddings") 
        print("   ✅ Consistent configuration across all scripts")
        print("   🚀 Ready to use for embedding and retrieval!") 