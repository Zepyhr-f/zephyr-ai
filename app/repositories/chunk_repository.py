from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.retrieval.service import SemanticSearchResult


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search_similar(
        self,
        vector: list[float],
        limit: int,
    ) -> list[SemanticSearchResult]:
        distance = Chunk.content_vector.cosine_distance(vector).label("distance")
        statement = (
            select(
                Chunk.content,
                Chunk.header_path,
                Chunk.metadata_,
                Document.title,
                Document.file_path,
                distance,
            )
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance)
            .limit(limit)
        )

        rows = (await self.session.execute(statement)).all()
        return [
            SemanticSearchResult(
                content=row.content,
                header_path=row.header_path,
                document_title=row.title,
                document_file_path=row.file_path,
                metadata=row.metadata_,
                distance=float(row.distance),
                score=1 - float(row.distance),
            )
            for row in rows
        ]
