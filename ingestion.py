"""Ingest annual report PDFs into embedding-ready JSON records.

LlamaParse is used to preserve the structure of financial tables as Markdown.
Text is chunked independently, while each detected Markdown table is kept as a
single record so that row and column relationships are not lost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedBlock:
    """A page-level text or table block extracted from parsed Markdown."""

    content: str
    page_number: int
    block_type: str


class FinancialReportParser:
    """Parse annual report PDFs and serialize text/table records to JSON."""

    _TABLE_SEPARATOR = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )

    def __init__(
        self,
        input_dir: str | Path = "data/raw",
        output_dir: str | Path = "data/processed",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize paths, the text splitter, and the LlamaParse client.

        Args:
            input_dir: Directory containing source PDF files.
            output_dir: Directory where the processed JSON file is written.
            chunk_size: Maximum number of characters in a text chunk.
            chunk_overlap: Number of overlapping characters between text chunks.

        Raises:
            ValueError: If the chunking configuration is invalid.
            RuntimeError: If ``LLAMA_CLOUD_API_KEY`` is not configured.
        """
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and larger than chunk_overlap")

        load_dotenv()
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is missing. Add it to .env or the environment."
            )

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
        )

    async def parse_pdf(self, pdf_path: Path) -> list[ParsedBlock]:
        """Parse one PDF with LlamaParse and separate text from table blocks.

        Args:
            pdf_path: Path to the PDF to parse.

        Returns:
            Page-aware blocks whose tables are complete Markdown strings.

        Raises:
            RuntimeError: If LlamaParse cannot parse the PDF.
        """
        try:
            documents = await self.parser.aload_data(str(pdf_path))
        except Exception as exc:
            LOGGER.exception("Failed to parse PDF '%s'.", pdf_path)
            raise RuntimeError(f"Unable to parse PDF: {pdf_path}") from exc

        blocks: list[ParsedBlock] = []
        for index, document in enumerate(documents, start=1):
            metadata: dict[str, Any] = getattr(document, "metadata", {}) or {}
            page_number = self._page_number(metadata, index)
            page_blocks = self._separate_markdown_blocks(
                str(getattr(document, "text", document)), page_number
            )
            blocks.extend(page_blocks)
        return blocks

    def process_blocks(
        self, blocks: Iterable[ParsedBlock], source_file: str
    ) -> list[dict[str, Any]]:
        """Chunk text blocks and preserve each table as one output record.

        Args:
            blocks: Parsed page-level text and table blocks.
            source_file: Name of the source PDF for output metadata.

        Returns:
            JSON-serializable records ready for embedding.
        """
        records: list[dict[str, Any]] = []
        for block in blocks:
            if block.block_type == "table":
                records.append(self._record(block.content, source_file, block))
                continue

            for chunk in self.text_splitter.split_text(block.content):
                records.append(self._record(chunk, source_file, block))
        return records

    async def process_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Parse and transform one PDF into embedding-ready records."""
        blocks = await self.parse_pdf(pdf_path)
        return self.process_blocks(blocks, pdf_path.name)

    async def process_all(self, output_filename: str = "processed_data.json") -> Path:
        """Process every PDF in the input directory and write one JSON file.

        Args:
            output_filename: Name of the JSON file within ``output_dir``.

        Returns:
            The path of the written JSON file.

        Raises:
            FileNotFoundError: If the input directory does not exist or has no PDFs.
        """
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        pdf_paths = sorted(self.input_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in: {self.input_dir}")

        all_records: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            LOGGER.info("Processing '%s'.", pdf_path)
            try:
                all_records.extend(await self.process_pdf(pdf_path))
            except RuntimeError:
                LOGGER.error("Skipping '%s' after parsing failure.", pdf_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_filename
        output_path.write_text(
            json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Wrote %d records to '%s'.", len(all_records), output_path)
        return output_path

    def _separate_markdown_blocks(
        self, markdown: str, page_number: int
    ) -> list[ParsedBlock]:
        """Split page Markdown around tables identified by header separators."""
        lines = markdown.splitlines()
        blocks: list[ParsedBlock] = []
        text_lines: list[str] = []
        index = 0

        def flush_text() -> None:
            content = "\n".join(text_lines).strip()
            if content:
                blocks.append(ParsedBlock(content, page_number, "text"))
            text_lines.clear()

        while index < len(lines):
            if index + 1 < len(lines) and self._is_table_header(lines[index], lines[index + 1]):
                flush_text()
                table_lines = [lines[index], lines[index + 1]]
                index += 2
                while index < len(lines) and self._is_table_row(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                blocks.append(ParsedBlock("\n".join(table_lines), page_number, "table"))
                continue
            text_lines.append(lines[index])
            index += 1

        flush_text()
        return blocks

    @classmethod
    def _is_table_header(cls, header: str, separator: str) -> bool:
        """Return whether two Markdown lines form a table header."""
        return "|" in header and cls._TABLE_SEPARATOR.match(separator) is not None

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Return whether a line belongs to a Markdown table."""
        return "|" in line and bool(line.strip())

    @staticmethod
    def _page_number(metadata: dict[str, Any], fallback: int) -> int:
        """Extract a page number from LlamaParse metadata with a safe fallback."""
        value = metadata.get("page_number", metadata.get("page_label", fallback))
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _record(
        content: str, source_file: str, block: ParsedBlock
    ) -> dict[str, Any]:
        """Build a consistently shaped output record."""
        return {
            "text": content,
            "metadata": {
                "source_file": source_file,
                "page_number": block.page_number,
                "type": block.block_type,
            },
        }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the ingestion script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-file", default="processed_data.json")
    return parser.parse_args()


def main() -> None:
    """Run PDF ingestion from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    try:
        output_path = asyncio.run(
            FinancialReportParser(args.input_dir, args.output_dir).process_all(
                args.output_file
            )
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        LOGGER.error("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc
    LOGGER.info("Ingestion complete: %s", output_path)


if __name__ == "__main__":
    main()