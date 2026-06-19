import uuid

from sqlalchemy import func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.markdown import MarkdownChunk, MarkdownDocument
from app.ingestion.service import StoredDocumentChunk
from app.models import Chunk, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_document_with_chunks(
        self,
        document: MarkdownDocument,
        chunk_vectors: list[tuple[MarkdownChunk, list[float]]],
    ) -> list[StoredDocumentChunk]:
        stored_document = await self.session.scalar(
            select(Document).where(Document.file_path == document.file_path)
        )
        if stored_document is None:
            stored_document = Document(
                file_path=document.file_path,
                title=document.title,
                metadata_=document.metadata,
            )
            self.session.add(stored_document)
            await self.session.flush()
        else:
            stored_document.title = document.title
            stored_document.metadata_ = document.metadata
            await self.session.execute(
                delete(Chunk).where(Chunk.document_id == stored_document.id)
            )

        for chunk, vector in chunk_vectors:
            self.session.add(
                Chunk(
                    document=stored_document,
                    content=chunk.content,
                    content_vector=vector,
                    chunk_index=chunk.chunk_index,
                    header_path=chunk.header_path,
                    metadata_=chunk.metadata,
                )
            )

        await self.session.flush()
        return [
            StoredDocumentChunk(chunk=chunk, vector=vector)
            for chunk, vector in chunk_vectors
        ]

    async def count(self) -> int:
        return int(await self.session.scalar(select(func.count()).select_from(Document)) or 0)

    async def list_documents(self, *, offset: int = 0, limit: int = 50) -> list[Document]:
        result = await self.session.execute(
            select(Document).order_by(Document.updated_at.desc()).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)
