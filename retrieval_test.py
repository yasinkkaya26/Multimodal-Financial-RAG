import os 
import psycopg2
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.getenv("DATABASE_URL")

model = SentenceTransformer("intfloat/multilingual-e5-small")

query = "Tüpraş'ın 2025 yılında toplam geliri ne kadardı?"

query_embedding = model.encode(
    f"query: {query}",
    normalize_embeddings=True
)

vector_string = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

sql = """
SELECT
    content,
    metadata,
    1 - (embedding <=> %s::vector) AS similarity
FROM document_chunks
ORDER BY embedding <=> %s::vector
LIMIT 5;
"""

with psycopg2.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(sql, (vector_string, vector_string))
        results = cur.fetchall()

for i, (content, metadata, similarity) in enumerate(results, 1):
    print(f"\n--- RESULT {i} ---")
    print(f"Similarity: {similarity:.4f}")
    print(f"Metadata: {metadata}")
    print(content[:1000])