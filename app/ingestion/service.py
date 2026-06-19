from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.embedding_client import EmbeddingClient
from app.core.vector_utils import normalize_vector
from app.ingestion.markdown import (
    MarkdownChunk,
    MarkdownDocument,
    discover_markdown_files,
    load_markdown_file,
    split_markdown_document,
)


DEFAULT_DOCS_ROOT = Path("/home/ubuntu/workspace/zephyr-docs")


class EmbeddingIngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredDocumentChunk:
    chunk: MarkdownChunk
    vector: list[float]


@dataclass(frozen=True)
class DocumentIngestionResult:
    document: MarkdownDocument
    chunk_count: int
    stored_chunks: list[StoredDocumentChunk]


@dataclass(frozen=True)
class DirectoryIngestionResult:
    root: str
    document_count: int
    chunk_count: int
    results: list[DocumentIngestionResult]


class DocumentIngestionStore(Protocol):
    async def upsert_document_with_chunks(
        self,
        document: MarkdownDocument,
        chunk_vectors: list[tuple[MarkdownChunk, list[float]]],
    ) -> list[StoredDocumentChunk]: ...


class EmbeddingIngestionService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        store: DocumentIngestionStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.store = store

    async def ingest_document(
        self,
        document: MarkdownDocument,
        chunks: list[MarkdownChunk],
    ) -> DocumentIngestionResult:
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embedding_client.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise EmbeddingIngestionError(
                f"Embedding vector count {len(vectors)} does not match chunk count {len(chunks)}"
            )

        try:
            normalized_vectors = [normalize_vector(vector) for vector in vectors]
        except ValueError as exc:
            raise EmbeddingIngestionError("Embedding vector normalization failed") from exc

        chunk_vectors = list(zip(chunks, normalized_vectors))
        stored_chunks = await self.store.upsert_document_with_chunks(document, chunk_vectors)
        return DocumentIngestionResult(
            document=document,
            chunk_count=len(chunks),
            stored_chunks=stored_chunks,
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> DocumentIngestionResult:
        document = load_markdown_file(path)
        chunks = split_markdown_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return await self.ingest_document(document, chunks)

    async def ingest_directory(
        self,
        root: str | Path = DEFAULT_DOCS_ROOT,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> DirectoryIngestionResult:
        paths = sorted(discover_markdown_files(root))
        results: list[DocumentIngestionResult] = []
        for path in paths:
            results.append(
                await self.ingest_file(
                    path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

        return DirectoryIngestionResult(
            root=str(root),
            document_count=len(results),
            chunk_count=sum(result.chunk_count for result in results),
            results=results,
        )
