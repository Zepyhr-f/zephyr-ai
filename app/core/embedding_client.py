from typing import Protocol

import httpx

from app.core.config import Settings


class EmbeddingClient(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class EmbeddingProviderError(RuntimeError):
    pass


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http_client = http_client or httpx.AsyncClient()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self._post({"model": self.model, "input": texts})
        return self._extract_embeddings(response)

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        if len(embeddings) != 1:
            raise EmbeddingProviderError("Embedding provider returned an unexpected number of vectors")
        return embeddings[0]

    async def _post(self, payload: dict) -> dict:
        try:
            response = await self.http_client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise EmbeddingProviderError(
                f"Embedding provider returned HTTP {status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError("Embedding provider request failed") from exc
        except ValueError as exc:
            raise EmbeddingProviderError("Embedding provider returned invalid JSON") from exc

    def _extract_embeddings(self, response: dict) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list):
            raise EmbeddingProviderError("Embedding provider response missing data list")

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingProviderError("Embedding provider response has invalid embedding data")
            vector = item["embedding"]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
                raise EmbeddingProviderError("Embedding provider response has non-numeric vector values")
            embeddings.append([float(value) for value in vector])
        return embeddings


ArkEmbeddingClient = OpenAICompatibleEmbeddingClient


def create_embedding_client(
    settings: Settings,
    http_client: httpx.AsyncClient | None = None,
) -> EmbeddingClient:
    if settings.embedding_provider != "ark":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    if not settings.ark_embedding_api_key:
        raise ValueError("ARK_EMBEDDING_API_KEY is required")

    return ArkEmbeddingClient(
        api_key=settings.ark_embedding_api_key,
        base_url=settings.ark_embedding_base_url,
        model=settings.ark_embedding_model,
        http_client=http_client,
    )
