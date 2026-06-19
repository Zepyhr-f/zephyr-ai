import argparse
import asyncio
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.embedding_client import create_embedding_client
from app.ingestion.service import DEFAULT_DOCS_ROOT, EmbeddingIngestionService
from app.repositories.document_repository import DocumentRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Markdown documents into pgvector.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_DOCS_ROOT),
        help="Markdown docs root directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


async def ingest_docs(root: str, chunk_size: int, chunk_overlap: int) -> None:
    started = time.monotonic()
    settings = get_settings()
    embedding_client = create_embedding_client(settings)

    async with AsyncSessionLocal() as session:
        service = EmbeddingIngestionService(
            embedding_client=embedding_client,
            store=DocumentRepository(session),
        )
        result = await service.ingest_directory(
            Path(root),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        await session.commit()

    elapsed = time.monotonic() - started
    print(f"root={result.root}")
    print(f"documents={result.document_count}")
    print(f"chunks={result.chunk_count}")
    print(f"duration_seconds={elapsed:.2f}")


def main() -> None:
    args = parse_args()
    asyncio.run(ingest_docs(args.root, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
