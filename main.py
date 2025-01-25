import sqlite3
import sqlite_vec
from openai import OpenAI
import os
import json
from typing import Callable, Tuple, List

def get_openai_embedding(word: str) -> Tuple[str, List[float]]:
    """Generate an embedding using OpenAI's API."""
    client = OpenAI()
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=word
    ).data[0].embedding
    return word, embedding

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the database with required tables and views."""
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Create tables
    db.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE
        )
    """)
    
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS word_embeddings USING vec0(
            embedding float[1536]
        )
    """)

    db.commit()
    return db

def find_similar_words(db: sqlite3.Connection, 
                      query_word: str, 
                      embedding_fn: Callable[[str], Tuple[str, List[float]]],
                      limit: int = 10) -> List[Tuple[str, float]]:
    """Find similar words to the query word using vector similarity."""
    # Generate embedding for query word
    _, query_embedding = embedding_fn(query_word)
    
    # Find similar words
    similar_words = db.execute("""
    SELECT 
        w.word,
        e.distance
    FROM (
        SELECT 
            rowid,
            distance
        FROM word_embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    ) e
    JOIN words w ON w.id = e.rowid
""", (json.dumps(query_embedding), limit)).fetchall()
    
    return similar_words

def store_embeddings(db: sqlite3.Connection, 
                    words: List[str],
                    embedding_fn: Callable[[str], Tuple[str, List[float]]]):
    """Store embeddings in SQLite."""
    for i, word in enumerate(words, 1):
        word, embedding = embedding_fn(word)
        # Store the word
        db.execute("INSERT OR IGNORE INTO words (id, word) VALUES (?, ?)", (i, word))
        # Store the embedding
        embedding_json = json.dumps(embedding)
        db.execute("INSERT OR REPLACE INTO word_embeddings(rowid, embedding) VALUES (?, ?)", 
                  (i, embedding_json))
    db.commit()

def query_word(db: sqlite3.Connection, query_word: str, limit: int = 10):
    """Query the database for similar words and print results."""
    results = find_similar_words(db, query_word, get_openai_embedding, limit)
    
    print(f"\nProximity search results for '{query_word}':")
    for word, distance in results:
        similarity = 1 / (1 + distance)  # Convert distance to similarity score
        print(f"{word}:\t{similarity:.4f} | {distance:.4f}")

if __name__ == "__main__":
    db = init_db("one.db")

    words = ["king", "queen", "man", "woman"]
    store_embeddings(db, words, get_openai_embedding)
    query_word(db, "child", limit=5)
    
    db.close()