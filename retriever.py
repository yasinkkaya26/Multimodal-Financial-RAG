import os
import re

import psycopg2
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


load_dotenv()


class FinancialRetriever:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")

        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured.")

        self.embedding_model = SentenceTransformer(
            self.EMBEDDING_MODEL
        )

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.RERANKER_MODEL
        )

        self.reranker = (
            AutoModelForSequenceClassification
            .from_pretrained(self.RERANKER_MODEL)
            .to(self.device)
        )

        self.reranker.eval()

    def _embed_query(self, query: str):
        embedding = self.embedding_model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )

        return "[" + ",".join(
            map(str, embedding.tolist())
        ) + "]"

    def _retrieve_candidates(
        self,
        query: str,
        candidate_limit: int = 20,
    ):
        query_vector = self._embed_query(query)

        lexical_query = self._build_lexical_query(query)
        print("lexical query:", lexical_query)

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

        LIMIT %s;
        """

        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        query_vector,
                        query_vector,
                        lexical_query,
                        lexical_query,
                        lexical_query,
                        candidate_limit,
                    ),
                )

                return cur.fetchall()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        candidates = self._retrieve_candidates(query)

        if not candidates:
            return []

        pairs = [
            [query, candidate[0]]
            for candidate in candidates
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            scores = (
                self.reranker(
                    **inputs,
                    return_dict=True,
                )
                .logits
                .view(-1)
                .float()
                .cpu()
                .tolist()
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for candidate, score in ranked[:top_k]:
            content, metadata, vector_rank, text_rank, rrf_score = candidate

            results.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "reranker_score": score,
                    "vector_rank": vector_rank,
                    "text_rank": text_rank,
                    "rrf_score": float(rrf_score),
                }
            )

        return results

    def _build_lexical_query(self, query: str) -> str:
        query_lower = query.lower()

        groups = []

        # Şirket
        if "tüpraş" in query_lower:
            groups.append("tüpraş")

        # Yıl
        import re

        years = re.findall(r"\b20\d{2}\b", query_lower)

        for year in years:
            groups.append(year)

        # Finansal kavramlar
        if "gelir" in query_lower or "ciro" in query_lower:
            groups.append("(gelir | ciro | cirosu | hasılat | satış)")

        if "kâr" in query_lower or "kar" in query_lower:
            groups.append("(kâr | kar | kazanç)")

        if "favök" in query_lower or "favok" in query_lower:
            groups.append("(favök | favok | ebitda)")

        if "borç" in query_lower:
            groups.append("(borç | borçlanma | yükümlülük)")

        # Hiçbir özel pattern yakalanmazsa raw query yerine
        # kelimeleri OR ile gevşek şekilde ara.
        if not groups:
            words = re.findall(r"\w+", query_lower)

            if words:
                groups.append(
                    "(" + " | ".join(words) + ")"
                )

        return " & ".join(groups)

        