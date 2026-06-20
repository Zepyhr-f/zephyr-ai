"""Tests for the Gateway HMAC sign middleware."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Iterable

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core import gateway_sign as gw_sign


def _user_to_dict(user):
    if user is None:
        return None
    return {
        "user_id": user.user_id,
        "username": user.username,
        "tenant_id": user.tenant_id,
        "roles": list(user.roles),
    }


class _StubStore:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    async def reserve(self, nonce: str, ttl_seconds: int) -> bool:
        if nonce in self.seen:
            return False
        self.seen.add(nonce)
        return True


def _make_app(secret: str, *, exempt: Iterable[str] = ("/health",), required: bool = True):
    from app.core.config import Settings, get_settings

    def settings_override() -> Settings:
        return Settings(
            zephyr_gateway_sign_secret=secret,
            zephyr_gateway_sign_required=required,
            zephyr_gateway_sign_ttl_seconds=300,
            zephyr_gateway_health_exempt=",".join(exempt),
        )

    # Force a fresh settings cache per test.
    get_settings.cache_clear()
    monkeypatched = settings_override()

    # Patch get_settings so the middleware sees the deterministic settings.
    original = gw_sign.get_settings
    gw_sign.get_settings = lambda: monkeypatched  # type: ignore[assignment]

    app = FastAPI()
    app.add_middleware(gw_sign.GatewaySignMiddleware)
    # Replace the (real) Redis-backed store with the in-memory stub.
    for mw in app.user_middleware:
        if mw.cls is gw_sign.GatewaySignMiddleware:
            break

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/ai/echo")
    async def echo(req: Request):  # type: ignore[no-redef]
        u = getattr(req.state, "user", None)
        return {"user": _user_to_dict(u)}

    @app.post("/api/v1/ai/echo")
    async def echo_post(req: Request):  # type: ignore[no-redef]
        u = getattr(req.state, "user", None)
        return {"user": _user_to_dict(u)}

    yielded = (app, monkeypatched, original)
    return yielded


def _sign(secret: str, method: str, path: str, ts: int, nonce: str, body: bytes = b"") -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    msg = "\n".join((method.upper(), path, str(ts), nonce, body_hash))
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


@pytest.fixture()
def app_factory(monkeypatch):
    created: list = []

    def _factory(secret: str = "unit-secret", **kwargs):
        app, settings_obj, original_get = _make_app(secret, **kwargs)
        # Inject stub store into the middleware after instantiation by walking ASGI stack.
        from app.core.gateway_sign import _NonceStore  # noqa: WPS437

        for mw in app.user_middleware:
            if mw.cls is gw_sign.GatewaySignMiddleware:
                kwargs_dict = mw.kwargs
                # Marker: mw is the BaseHTTPMiddleware factory; we need to monkeypatch
                # the dispatcher after build. We do that by overriding
                # GatewaySignMiddleware._store via subclassing instead.
        # Simplest: monkeypatch _NonceStore to return stub.

        store = _StubStore()
        monkeypatch.setattr(gw_sign, "_NonceStore", lambda **kw: store)
        # Rebuild app so middleware constructor uses patched store.
        app2, _, _ = _make_app(secret, **kwargs)
        created.append((app2, store, original_get))
        return app2, store

    yield _factory

    for _, _, original_get in created:
        gw_sign.get_settings = original_get  # type: ignore[assignment]
    gw_sign.get_settings.cache_clear() if hasattr(gw_sign.get_settings, "cache_clear") else None


def test_health_is_exempt(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_missing_signature_rejected(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    r = client.get("/api/v1/ai/echo")
    assert r.status_code == 401
    assert r.json()["msg"] == "gateway_sign_missing"


def test_valid_signature_accepted(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    secret = "unit-secret"
    ts = int(time.time())
    nonce = "n-ok"
    sig = _sign(secret, "GET", "/api/v1/ai/echo", ts, nonce)
    r = client.get(
        "/api/v1/ai/echo",
        headers={
            "x-zephyr-gateway-sign": sig,
            "x-zephyr-gateway-ts": str(ts),
            "x-zephyr-gateway-nonce": nonce,
            "x-zephyr-user-id": "u-1",
            "x-zephyr-username": "alice",
            "x-zephyr-roles": "admin,sysop",
        },
    )
    assert r.status_code == 200
    user = r.json()["user"]
    assert user["user_id"] == "u-1"
    assert user["username"] == "alice"
    assert user["roles"] == ["admin", "sysop"] or tuple(user["roles"]) == ("admin", "sysop")


def test_replay_rejected(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    secret = "unit-secret"
    ts = int(time.time())
    nonce = "n-replay"
    sig = _sign(secret, "GET", "/api/v1/ai/echo", ts, nonce)
    headers = {
        "x-zephyr-gateway-sign": sig,
        "x-zephyr-gateway-ts": str(ts),
        "x-zephyr-gateway-nonce": nonce,
    }
    r1 = client.get("/api/v1/ai/echo", headers=headers)
    r2 = client.get("/api/v1/ai/echo", headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 409
    assert r2.json()["msg"] == "gateway_sign_replay"


def test_expired_timestamp_rejected(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    secret = "unit-secret"
    ts = int(time.time()) - 3600
    nonce = "n-expired"
    sig = _sign(secret, "GET", "/api/v1/ai/echo", ts, nonce)
    r = client.get(
        "/api/v1/ai/echo",
        headers={
            "x-zephyr-gateway-sign": sig,
            "x-zephyr-gateway-ts": str(ts),
            "x-zephyr-gateway-nonce": nonce,
        },
    )
    assert r.status_code == 401
    assert r.json()["msg"] == "gateway_sign_expired"


def test_signature_mismatch_rejected(app_factory):
    app, _ = app_factory()
    client = TestClient(app)
    ts = int(time.time())
    r = client.get(
        "/api/v1/ai/echo",
        headers={
            "x-zephyr-gateway-sign": "deadbeef",
            "x-zephyr-gateway-ts": str(ts),
            "x-zephyr-gateway-nonce": "n-bad",
        },
    )
    assert r.status_code == 401
    assert r.json()["msg"] == "gateway_sign_invalid"
