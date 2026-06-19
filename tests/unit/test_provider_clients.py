import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import httpx
import pytest

from app.core.config import Settings


class FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200, text: str = "") -> None:
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("POST", "https://provider.example")
        self.headers = {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
                text=self.text,
                headers=self.headers,
            )
            raise httpx.HTTPStatusError("provider error", request=self.request, response=response)


class FakeAsyncHTTPClient:
    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response or FakeResponse()
        self.exc = exc
        self.calls: list[dict] = []

    async def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.exc:
            raise self.exc
        return self.response


@pytest.mark.parametrize(
    "env_name,env_value,attr",
    [
        ("LLM_PROVIDER", "anthropic", "llm_provider"),
        ("ANTHROPIC_AUTH_TOKEN", "test-token", "anthropic_auth_token"),
        ("ANTHROPIC_BASE_URL", "https://anthropic.example/api", "anthropic_base_url"),
        ("ANTHROPIC_MODEL", "ark-code-test", "anthropic_model"),
        ("EMBEDDING_PROVIDER", "ark", "embedding_provider"),
        ("ARK_EMBEDDING_API_KEY", "test-key", "ark_embedding_api_key"),
        ("ARK_EMBEDDING_BASE_URL", "https://ark.example/api/v3", "ark_embedding_base_url"),
        ("ARK_EMBEDDING_MODEL", "doubao-test", "ark_embedding_model"),
        ("LAS_API_KEY", "test-las-key", "las_api_key"),
        ("LAS_EMBEDDING_MODEL", "doubao-las-test", "las_embedding_model"),
        ("EMBEDDING_DIMENSION", "1024", "embedding_dimension"),
    ],
)
def test_settings_reads_provider_config(monkeypatch: pytest.MonkeyPatch, env_name, env_value, attr) -> None:
    monkeypatch.setenv(env_name, env_value)

    settings = Settings()

    expected = 1024 if attr == "embedding_dimension" else env_value
    assert getattr(settings, attr) == expected


def test_settings_provider_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_base_url == "https://ark.cn-beijing.volces.com/api/coding"
    assert settings.anthropic_model == "ark-code-latest"
    assert settings.embedding_provider == "ark"
    assert settings.ark_embedding_base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert settings.ark_embedding_model == "doubao-embedding"
    assert settings.las_embedding_model == "doubao-embedding"
    assert settings.embedding_dimension == 1536


@pytest.mark.asyncio
async def test_llm_client_posts_anthropic_messages_request() -> None:
    from app.core.llm_client import ChatMessage, create_llm_client

    http_client = FakeAsyncHTTPClient(
        FakeResponse({"content": [{"type": "text", "text": "hello"}]})
    )
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_auth_token="test-token",
        anthropic_base_url="https://anthropic.example/api",
        anthropic_model="ark-code-test",
    )

    client = create_llm_client(settings, http_client=http_client)
    text = await client.chat(
        [ChatMessage(role="user", content="Hi")],
        max_tokens=256,
        temperature=0.1,
    )

    assert text == "hello"
    assert http_client.calls == [
        {
            "url": "https://anthropic.example/api/v1/messages",
            "headers": {
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
            },
            "json": {
                "model": "ark-code-test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 256,
                "temperature": 0.1,
            },
        }
    ]


@pytest.mark.asyncio
async def test_llm_client_parses_multiple_text_blocks() -> None:
    from app.core.llm_client import ChatMessage, create_llm_client

    http_client = FakeAsyncHTTPClient(
        FakeResponse(
            {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "name": "ignored"},
                    {"type": "text", "text": " world"},
                ]
            }
        )
    )
    settings = Settings(_env_file=None, anthropic_auth_token="test-token")

    client = create_llm_client(settings, http_client=http_client)
    text = await client.chat([ChatMessage(role="user", content="Hi")])

    assert text == "hello world"


@pytest.mark.asyncio
async def test_embedding_client_posts_openai_compatible_request() -> None:
    from app.core.embedding_client import create_embedding_client

    http_client = FakeAsyncHTTPClient(
        FakeResponse({"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]})
    )
    settings = Settings(
        _env_file=None,
        embedding_provider="ark",
        ark_embedding_api_key="test-key",
        ark_embedding_base_url="https://ark.example/api/v3",
        ark_embedding_model="doubao-test",
    )

    client = create_embedding_client(settings, http_client=http_client)
    vectors = await client.embed_texts(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert http_client.calls == [
        {
            "url": "https://ark.example/api/v3/embeddings",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "json": {"model": "doubao-test", "input": ["alpha", "beta"]},
        }
    ]


@pytest.mark.asyncio
async def test_embedding_client_embed_query_returns_single_vector() -> None:
    from app.core.embedding_client import create_embedding_client

    http_client = FakeAsyncHTTPClient(FakeResponse({"data": [{"embedding": [0.1, 0.2]}]}))
    settings = Settings(_env_file=None, ark_embedding_api_key="test-key")

    client = create_embedding_client(settings, http_client=http_client)
    vector = await client.embed_query("alpha")

    assert vector == [0.1, 0.2]


@pytest.mark.asyncio
async def test_create_embedding_client_supports_las_doubao(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.embedding_client import LasDoubaoEmbeddingClient, create_embedding_client

    class FakeTable:
        def __init__(self, data: dict) -> None:
            self.data = data

        def __getitem__(self, key: str) -> list[str]:
            return self.data[key]

        def with_column(self, name: str, values: list[list[float]]) -> "FakeTable":
            self.data[name] = values
            return self

        def collect(self) -> "FakeTable":
            return self

        def to_pydict(self) -> dict:
            return self.data

    class FakeDaft(ModuleType):
        def from_pydict(self, data: dict) -> FakeTable:
            return FakeTable(data)

    class FakeDoubaoEmbeddingText:
        pass

    def fake_col(name: str) -> str:
        return name

    def fake_las_udf(function: object, construct_args: dict) -> object:
        assert function is FakeDoubaoEmbeddingText
        assert construct_args == {"model": "doubao-las-test"}

        def apply(_: str) -> list[list[float]]:
            return [[0.0, 1.0], [1.0, 2.0]]

        return apply

    fake_daft = FakeDaft("daft")
    fake_daft.col = fake_col
    fake_daft_las = ModuleType("daft.las")
    fake_daft_las_functions = ModuleType("daft.las.functions")
    fake_daft_las_functions_ark_llm = ModuleType("daft.las.functions.ark_llm")
    fake_doubao_module = ModuleType("daft.las.functions.ark_llm.doubao_embedding_text")
    fake_doubao_module.DoubaoEmbeddingText = FakeDoubaoEmbeddingText
    fake_udf_module = ModuleType("daft.las.functions.udf")
    fake_udf_module.las_udf = fake_las_udf
    monkeypatch.setitem(sys.modules, "daft", fake_daft)
    monkeypatch.setitem(sys.modules, "daft.las", fake_daft_las)
    monkeypatch.setitem(sys.modules, "daft.las.functions", fake_daft_las_functions)
    monkeypatch.setitem(sys.modules, "daft.las.functions.ark_llm", fake_daft_las_functions_ark_llm)
    monkeypatch.setitem(sys.modules, "daft.las.functions.ark_llm.doubao_embedding_text", fake_doubao_module)
    monkeypatch.setitem(sys.modules, "daft.las.functions.udf", fake_udf_module)

    settings = Settings(
        _env_file=None,
        embedding_provider="las_doubao",
        las_embedding_model="doubao-las-test",
    )
    settings.las_api_key = "test-las-token"

    client = create_embedding_client(settings)
    vectors = await client.embed_texts(["alpha", "beta"])

    assert isinstance(client, LasDoubaoEmbeddingClient)
    assert vectors == [[0.0, 1.0], [1.0, 2.0]]


def test_las_doubao_missing_api_key_raises() -> None:
    from app.core.embedding_client import create_embedding_client

    settings = Settings(_env_file=None, embedding_provider="las_doubao")

    with pytest.raises(ValueError, match="LAS_API_KEY is required"):
        create_embedding_client(settings)


@pytest.mark.asyncio
async def test_las_doubao_error_message_does_not_leak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.embedding_client import EmbeddingProviderError, create_embedding_client

    non_sensitive_token = "test-las-token-redacted"

    def fake_load_las_dependencies():
        raise ImportError("missing dependency")

    monkeypatch.setattr("app.core.embedding_client._load_las_dependencies", fake_load_las_dependencies)
    settings = Settings(_env_file=None, embedding_provider="las_doubao")
    settings.las_api_key = non_sensitive_token
    client = create_embedding_client(settings)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await client.embed_texts(["alpha"])

    assert non_sensitive_token not in str(exc_info.value)


def test_unsupported_providers_raise_clear_errors() -> None:
    from app.core.embedding_client import create_embedding_client
    from app.core.llm_client import create_llm_client

    llm_settings = cast(Settings, SimpleNamespace(llm_provider="unsupported"))
    embedding_settings = cast(Settings, SimpleNamespace(embedding_provider="unsupported"))

    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        create_llm_client(llm_settings)
    with pytest.raises(ValueError, match="Unsupported embedding provider: unsupported"):
        create_embedding_client(embedding_settings)


@pytest.mark.asyncio
async def test_llm_error_message_does_not_leak_token() -> None:
    from app.core.llm_client import ChatMessage, LLMProviderError, create_llm_client

    secret = "secret-token-123"
    http_client = FakeAsyncHTTPClient(FakeResponse(status_code=401, text=f"bad token {secret}"))
    settings = Settings(_env_file=None, anthropic_auth_token=secret)

    client = create_llm_client(settings, http_client=http_client)

    with pytest.raises(LLMProviderError) as exc_info:
        await client.chat([ChatMessage(role="user", content="Hi")])

    assert secret not in str(exc_info.value)


@pytest.mark.asyncio
async def test_embedding_error_message_does_not_leak_key() -> None:
    from app.core.embedding_client import EmbeddingProviderError, create_embedding_client

    secret = "secret-key-123"
    http_client = FakeAsyncHTTPClient(FakeResponse(status_code=403, text=f"bad key {secret}"))
    settings = Settings(_env_file=None, ark_embedding_api_key=secret)

    client = create_embedding_client(settings, http_client=http_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await client.embed_texts(["alpha"])

    assert secret not in str(exc_info.value)
