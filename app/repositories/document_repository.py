from sqlalchemy import delete, select
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
