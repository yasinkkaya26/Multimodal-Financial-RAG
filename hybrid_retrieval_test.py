import os
import psycopg2
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATABASE_URL = os.getenv("DATABASE_URL")

model = SentenceTransformer("intfloat/multilingual-e5-small")

query = "Tüpraş'ın 2025 yılında toplam geliri ne kadardı?"

# Şimdilik test amaçlı finansal keyword expansion.
# Daha sonra bunu otomatik hale getireceğiz.
lexical_query = "tüpraş & 2025 & (gelir | ciro | hasılat | satış)"

query_embedding = model.encode(
    f"query: {query}",
    normalize_embeddings=True,
)

vector_string = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

sql = """
WITH vector_results AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY embedding <=> %s::vector
        ) AS vector_rank
    FROM document_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT 30
),

text_results AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(
                to_tsvector('simple', content),
                to_tsquery('simple', %s)
            ) DESC
        ) AS text_rank
    FROM document_chunks
    WHERE
        to_tsvector('simple', content)
        @@ to_tsquery('simple', %s)
    ORDER BY
        ts_rank_cd(
            to_tsvector('simple', content),
            to_tsquery('simple', %s)
        ) DESC
    LIMIT 30
),

combined AS (
    SELECT
        COALESCE(v.id, t.id) AS id,
        v.vector_rank,
        t.text_rank,

        COALESCE(
            1.0 / (60 + v.vector_rank),
            0
        )
        +
        COALESCE(
            1.0 / (60 + t.text_rank),
            0
        ) AS rrf_score

    FROM vector_results v

    FULL OUTER JOIN text_results t
        ON v.id = t.id
)

SELECT
    d.content,
    d.metadata,
    c.vector_rank,
    c.text_rank,
    c.rrf_score

FROM combined c

JOIN document_chunks d
    ON d.id = c.id

ORDER BY c.rrf_score DESC

LIMIT 20;
"""

with psycopg2.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                vector_string,
                vector_string,
                lexical_query,
                lexical_query,
                lexical_query,
            ),
        )

        results = cur.fetchall()

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        reranker_name = "BAAI/bge-reranker-v2-m3"

        tokenizer = AutoTokenizer.from_pretrained(reranker_name)

        reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name)

        reranker.to(device)
        reranker.eval()

        pairs = [
            [query, result[0]]
            for result in results
        ]

        inputs = tokenizer(
            pairs,
            padding = True,
            truncation = True,
            return_tensors = "pt",
            max_length = 512,
        )

        inputs = {
            key: value.to(device)
            for key,value in inputs.items()
        }

        with torch.no_grad():
            scores = reranker(
                **inputs,
                return_dict=True
            ).logits.view(-1).float().cpu().tolist()

        reranked_results = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )




for i, (result, reranker_score) in enumerate(
    reranked_results[:5],
    1
):
    content, metadata, vector_rank, text_rank, rrf_score = result

    print(f"\n--- RESULT {i} ---")
    print(f"Reranker score: {reranker_score:.4f}")
    print(f"Vector rank: {vector_rank}")
    print(f"Text rank: {text_rank}")
    print(f"RRF score: {rrf_score:.6f}")
    print(f"Metadata: {metadata}")
    print(content[:1000])