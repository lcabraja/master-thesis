import sqlite3
import sqlite_vec
from openai import OpenAI
import os
import json

# Initialize OpenAI client
client = OpenAI()

# Connect to SQLite and load vector extension
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Create a virtual table for word embeddings
db.execute("""
    CREATE VIRTUAL TABLE word_embeddings USING vec0(
        embedding float[1536]
    )
""")

# Generate and store embeddings
words = ["king", "queen", "man", "woman"]
word_to_id = {}

for i, word in enumerate(words, 1):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=word
    ).data[0].embedding
    # Store the word-to-id mapping
    word_to_id[word] = i
    # Convert embedding to JSON string
    embedding_json = json.dumps(embedding)
    db.execute("INSERT INTO word_embeddings(rowid, embedding) VALUES (?, ?)", 
               (i, embedding_json))

db.commit()

# Get woman's embedding
woman_embedding = db.execute("""
    SELECT embedding
    FROM word_embeddings
    WHERE rowid = ?
""", (word_to_id['woman'],)).fetchone()[0]

# Perform proximity search for 'woman'
woman_results = db.execute("""
    SELECT 
        rowid,
        distance
    FROM word_embeddings
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT 3
""", (woman_embedding,)).fetchall()

print("\nProximity search results for 'woman':")
for rowid, distance in woman_results:
    word = [w for w, id in word_to_id.items() if id == rowid][0]
    similarity = 1 / (1 + distance)  # Convert distance to similarity score
    print(f"{word}: {similarity:.4f}")