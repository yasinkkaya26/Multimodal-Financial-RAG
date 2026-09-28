"""Generate local embeddings and populate the PostgreSQL vector store."""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2 import sql
from psycopg2.extras import Json, execute_values, register_uuid
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)
register_uuid()


class VectorDBManager:
    """Manage embedding generation and persistence for parsed report chunks."""

    TABLE_NAME = "document_chunks"
    EMBEDDING_DIMENSION = 384
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDABLE_TYPES = frozenset(
        {"text", "markdown", "text/markdown", "table"}
    )

    def __init__(
        self,
        database_url: str | None = None,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> None:
        """Initialize the embedding model and database settings."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        load_dotenv()
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is missing from .env or the environment")

        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self._embedding_device())
        dimension = self.model.get_embedding_dimension()
        if dimension != self.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Model '{model_name}' produces {dimension}-dimension vectors; "
                f"this schema requires {self.EMBEDDING_DIMENSION}."
            )

    @staticmethod
    def _embedding_device() -> str:
        """Select Apple Metal acceleration when available, otherwise CPU."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for local embeddings") from exc

        if torch.backends.mps.is_available():
            LOGGER.info("Using Apple Metal Performance Shaders (MPS) for embeddings.")
            return "mps"

        LOGGER.warning("MPS is unavailable; falling back to CPU embeddings.")
        return "cpu"

    def create_schema(self) -> None:
        """Enable pgvector and create the document chunk table if needed."""
        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id UUID PRIMARY KEY,
                            content TEXT NOT NULL,
                            metadata JSONB NOT NULL,
                            embedding VECTOR({}) NOT NULL
                        )
                        """
                    ).format(
                        sql.Identifier(self.TABLE_NAME),
                        sql.Literal(self.EMBEDDING_DIMENSION),
                    )
                )
            register_vector(connection)
        LOGGER.info("Database schema is ready.")

    def load_records(self, input_path: Path) -> list[dict[str, Any]]:
        """Load and validate parsed records from a JSON file."""
        if not input_path.is_file():
            raise FileNotFoundError(f"Processed JSON file does not exist: {input_path}")

        try:
            with input_path.open("r", encoding="utf-8") as file:
                records = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in processed file: {input_path}") from exc

        if not isinstance(records, list):
            raise ValueError("Processed JSON must contain a list of records")
        return [record for record in records if self._is_embeddable(record)]

    def populate(self, input_path: Path) -> int:
        """Embed eligible records and insert them into PostgreSQL."""
        records = self.load_records(input_path)
        if not records:
            LOGGER.warning("No embeddable records found in '%s'.", input_path)
            return 0

        inserted = 0
        try:
            with psycopg2.connect(self.database_url) as connection:
                register_vector(connection)
                with connection.cursor() as cursor:
                    for batch in self._batches(records):
                        contents = [record["text"].strip() for record in batch]
                        embeddings = self.model.encode(
                            contents,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        rows = [
                            (
                                uuid.uuid4(),
                                record["text"].strip(),
                                Json(record["metadata"]),
                                embedding.tolist(),
                            )
                            for record, embedding in zip(batch, embeddings)
                        ]
                        execute_values(
                            cursor,
                            """
                            INSERT INTO document_chunks (id, content, metadata, embedding)
                            VALUES %s
                            """,
                            rows,
                            page_size=len(rows),
                        )
                        inserted += len(rows)
                        LOGGER.info("Inserted %d/%d chunks.", inserted, len(records))
        except psycopg2.Error:
            LOGGER.exception("Database operation failed; transaction rolled back.")
            raise

        return inserted

    @classmethod
    def _is_embeddable(cls, record: Any) -> bool:
        """Return whether a parsed record contains supported text content."""
        if not isinstance(record, dict):
            return False
        content = record.get("text")
        metadata = record.get("metadata")
        return (
            isinstance(content, str)
            and bool(content.strip())
            and isinstance(metadata, dict)
            and metadata.get("type") in cls.EMBEDDABLE_TYPES
        )

    def _batches(
        self, records: Sequence[dict[str, Any]]
    ) -> Iterator[Sequence[dict[str, Any]]]:
        """Yield records in fixed-size batches."""
        for start in range(0, len(records), self.batch_size):
            yield records[start : start + self.batch_size]


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed_data.json"),
        help="Path to the processed chunks JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Create the schema and populate the vector database."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    try:
        manager = VectorDBManager(batch_size=args.batch_size)
        manager.create_schema()
        inserted = manager.populate(args.input)
        LOGGER.info("Completed database population: %d chunks inserted.", inserted)
    except psycopg2.OperationalError:
        LOGGER.exception(
            "Could not connect to PostgreSQL. Start the database with "
            "'docker compose up -d db' and verify DATABASE_URL."
        )
        raise SystemExit(1)
    except (FileNotFoundError, RuntimeError, ValueError, psycopg2.Error):
        LOGGER.exception("Database population failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()