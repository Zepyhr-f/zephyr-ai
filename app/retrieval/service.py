from dataclasses import dataclass
from typing import Protocol

from app.core.embedding_client import EmbeddingClient
from app.core.vector_utils import normalize_vector


@dataclass(frozen=True)
class SemanticSearchResult:
    content: str
    header_path: str | None
    document_title: str | None
    document_file_path: str
    metadata: dict | None
    distance: float
    score: float
    chunk_id: str | None = None
    document_id: str | None = None


class SearchSimilarRepository(Protocol):
    async def search_similar(
        self,
        vector: list[float],
        limit: int,
    ) -> list[SemanticSearchResult]: ...


class SemanticRetrievalService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        repository: SearchSimilarRepository,
    ) -> None:
        self.embedding_client = embedding_client
        self.repository = repository

    async def search(self, query: str, limit: int = 5) -> list[SemanticSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        query_vector = await self.embedding_client.embed_query(normalized_query)
        normalized_vector = normalize_vector(query_vector)
        return await self.repository.search_similar(normalized_vector, limit)
