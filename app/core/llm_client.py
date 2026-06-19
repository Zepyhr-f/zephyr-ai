from dataclasses import asdict, dataclass
from typing import Protocol

import httpx

from app.core.config import Settings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str: ...


class LLMProviderError(RuntimeError):
    pass


class AnthropicCompatibleLLMClient:
    def __init__(
        self,
        *,
        auth_token: str,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.auth_token = auth_token
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http_client = http_client or httpx.AsyncClient()

    async def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._post(payload)
        return self._extract_text(response)

    async def _post(self, payload: dict) -> dict:
        try:
            response = await self.http_client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise LLMProviderError(f"LLM provider returned HTTP {status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("LLM provider request failed") from exc
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned invalid JSON") from exc

    def _extract_text(self, response: dict) -> str:
        content = response.get("content")
        if not isinstance(content, list):
            raise LLMProviderError("LLM provider response missing content list")

        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if not text_parts:
            raise LLMProviderError("LLM provider response did not include text content")
        return "".join(text_parts)


class OpenAICompatibleLLMClient:
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

    async def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._post(payload)
        return self._extract_text(response)

    async def _post(self, payload: dict) -> dict:
        try:
            response = await self.http_client.post(
                f"{self.base_url}/chat/completions",
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
            raise LLMProviderError(f"LLM provider returned HTTP {status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("LLM provider request failed") from exc
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned invalid JSON") from exc

    def _extract_text(self, response: dict) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("LLM provider response missing choices list")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMProviderError("LLM provider response choice is invalid")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError("LLM provider response missing message content")
        return message["content"]


def create_llm_client(settings: Settings, http_client: httpx.AsyncClient | None = None) -> LLMClient:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            http_client=http_client,
        )

    if settings.llm_provider != "anthropic":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    auth_token = settings.anthropic_auth_token or settings.anthropic_api_key
    if not auth_token:
        raise ValueError("ANTHROPIC_AUTH_TOKEN is required")

    return AnthropicCompatibleLLMClient(
        auth_token=auth_token,
        base_url=settings.anthropic_base_url,
        model=settings.anthropic_model,
        http_client=http_client,
    )
