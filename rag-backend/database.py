import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "embeddings.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    return conn

def save_embedding(doc_id, content, embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO documents (id, content, embedding) VALUES (?, ?, ?)",
        (doc_id, content, json.dumps(embedding))
    )
    conn.commit()
    conn.close()

def load_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    return [(doc_id, content, json.loads(embedding)) for doc_id, content, embedding in rows]
