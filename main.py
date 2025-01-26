import sqlite3
import sqlite_vec
from openai import OpenAI
import os
import json
import pandas as pd
from typing import Callable, Tuple, List
import time
import sys

def get_embedding(client: OpenAI, model: str, dimensions: int, word: str) -> Tuple[str, List[float], float, int]:
    """Generate an embedding using OpenAI's API."""
    start_time = time.time()
    response = client.embeddings.create(
        model=model,
        input=word,
        dimensions=dimensions
    )
    end_time = time.time()
    embedding = response.data[0].embedding
    
    # Calculate data size (approximate size of the word and the embedding)
    word_size = sys.getsizeof(word)
    embedding_size = sys.getsizeof(embedding)
    total_size = word_size + embedding_size
    
    return word, embedding, end_time - start_time, total_size

def init_db(db_path: str, dimensions: int) -> sqlite3.Connection:
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
    
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS word_embeddings USING vec0(
            embedding float[{dimensions}]
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
    results = find_similar_words(db, query_word, get_embedding, limit)
    
    print(f"\nProximity search results for '{query_word}':")
    for word, distance in results:
        similarity = 1 / (1 + distance)  # Convert distance to similarity score
        print(f"{word}:\t{similarity:.4f} | {distance:.4f}")

def process_movie_data(db: sqlite3.Connection, csv_path: str, client: OpenAI, model: str, dimensions: int, limit: int = 100):
    """Read movie data from CSV and create embeddings for movie overviews."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df.head(limit)
    
    log_file = f"{model}-{dimensions}.log"
    total_time = 0
    total_size = 0
    
    with open(log_file, 'w') as f:
        f.write(f"Movie Embedding Statistics for model: {model}, dimensions: {dimensions}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Index':<6} {'Time (s)':<10} {'Size (bytes)':<12} {'Input (bytes)':<12} {'Overview Length':<15}\n")
        f.write("-" * 80 + "\n")
        
        # Process each movie
        for i, row in df.iterrows():
            movie_id = int(i + 1)
            overview = str(row['overview'])
            input_size = sys.getsizeof(overview)
            
            # Store the movie overview
            db.execute("INSERT OR IGNORE INTO words (id, word) VALUES (?, ?)", 
                      (movie_id, overview))
            
            # Generate and store the embedding
            _, embedding, duration, data_size = get_embedding(client, model, dimensions, overview)
            embedding_json = json.dumps(embedding)
            db.execute("INSERT OR REPLACE INTO word_embeddings(rowid, embedding) VALUES (?, ?)", 
                      (movie_id, embedding_json))
            
            total_time += duration
            total_size += data_size
            
            # Log statistics
            f.write(f"{i:<6} {duration:<10.3f} {data_size:<12} {input_size:<12} {len(overview):<15}\n")
            f.flush()  # Ensure immediate writing to file
            
            if i % 10 == 0:
                db.commit()
                print(f"Processed {i+1} movies...")
        
        # Write summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
        f.write(f"Average Time per Movie: {total_time/len(df):.2f} seconds\n")
        f.write(f"Total Data Size: {total_size/1024:.2f} KB\n")
        f.write(f"Average Data Size per Movie: {total_size/len(df)/1024:.2f} KB\n")
    
    db.commit()
    print(f"Finished processing movies! Statistics saved to {log_file}")

if __name__ == "__main__":
    import sys
    
    model = "text-embedding-3-large"
    dimensions = 3072
    
    client = OpenAI()
    db = init_db(f"{model}-{dimensions}.db", dimensions)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Query mode: python main.py <query_text>")
        print("  Populate DB: python main.py populate-db [limit]")
        sys.exit(1)

    if sys.argv[1] == "populate-db":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        process_movie_data(db, "./tmdb_5000_movies.csv", client, model, dimensions, limit=limit)
    else:
        # Join all remaining arguments as the query text
        query = " ".join(sys.argv[1:])
        query_word(db, query, limit=5)
    
    db.close()

# headers
# budget,genres,homepage,id,keywords,original_language,original_title,overview,popularity,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,vote_average,vote_coun

