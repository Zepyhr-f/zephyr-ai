"""Gateway HMAC signature middleware for zephyr-ai.

Every external request to zephyr-ai must arrive through zephyr-gateway, which
attaches an HMAC-SHA256 signature, a timestamp and a nonce. This middleware
verifies all three and stores the nonce in Redis with a TTL so that replays are
rejected. Health checks are exempted.

The signing layout is identical to the Go side (see pkg/gatewaysign/signer.go):

    msg = METHOD + "\n" + path + "\n" + ts + "\n" + nonce + "\n" + sha256_hex(body)
    sig = hex(HMAC_SHA256(secret, msg))

When `ZEPHYR_GATEWAY_SIGN_REQUIRED=false` the middleware degrades to a noop —
useful only for local CLI smokes; production always keeps it on.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings


HEADER_SIGN = "x-zephyr-gateway-sign"
HEADER_TS = "x-zephyr-gateway-ts"
HEADER_NONCE = "x-zephyr-gateway-nonce"
HEADER_USER_ID = "x-zephyr-user-id"
HEADER_USERNAME = "x-zephyr-username"
HEADER_TENANT_ID = "x-zephyr-tenant-id"
HEADER_ROLES = "x-zephyr-roles"


@dataclass(slots=True)
class GatewayUser:
    user_id: str
    username: str | None
    tenant_id: str | None
    roles: tuple[str, ...]


def _canonical_message(method: str, path: str, ts: str, nonce: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    return "\n".join((method.upper(), path, ts, nonce, body_hash))


def _expected_signature(secret: str, message: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


class _NonceStore:
    """Async wrapper around redis.asyncio for replay protection."""

    def __init__(self, host: str, password: str | None, db: int, prefix: str):
        host_part, _, port_part = host.partition(":")
        port = int(port_part) if port_part else 6379
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError(
                "redis package missing; add `redis>=5` to requirements"
            ) from exc
        self._client = Redis(
            host=host_part,
            port=port,
            db=db,
            password=password or None,
            decode_responses=True,
        )
        self._prefix = prefix

    async def reserve(self, nonce: str, ttl_seconds: int) -> bool:
        ok = await self._client.set(
            f"{self._prefix}{nonce}",
            "1",
            nx=True,
            ex=ttl_seconds,
        )
        return bool(ok)

    async def aclose(self) -> None:  # pragma: no cover - housekeeping
        try:
            await self._client.aclose()
        except Exception:
            pass


class GatewaySignMiddleware(BaseHTTPMiddleware):
    """Validate gateway signature, populate request.state.user, then dispatch."""

    def __init__(self, app: ASGIApp, exempt_paths: Iterable[str] | None = None):
        super().__init__(app)
        settings = get_settings()
        self._secret = settings.zephyr_gateway_sign_secret
        self._ttl = max(1, int(settings.zephyr_gateway_sign_ttl_seconds))
        self._required = bool(settings.zephyr_gateway_sign_required)
        exempt = list(exempt_paths or [])
        if settings.zephyr_gateway_health_exempt:
            exempt.extend(
                p.strip()
                for p in settings.zephyr_gateway_health_exempt.split(",")
                if p.strip()
            )
        self._exempt = tuple(p.rstrip("/") or "/" for p in exempt)
        self._store: _NonceStore | None = None
        if self._required and self._secret:
            self._store = _NonceStore(
                host=settings.zephyr_gateway_redis_host,
                password=settings.zephyr_gateway_redis_password,
                db=settings.zephyr_gateway_redis_db,
                prefix=settings.zephyr_gateway_nonce_prefix,
            )

    def _is_exempt(self, path: str) -> bool:
        normalized = path.rstrip("/") or "/"
        for prefix in self._exempt:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return True
        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ):
        if not self._required:
            return await call_next(request)
        if self._is_exempt(request.url.path):
            return await call_next(request)
        if not self._secret:
            return JSONResponse(
                {"code": 500, "msg": "gateway_sign_secret_missing"}, status_code=500
            )

        sign = request.headers.get(HEADER_SIGN, "")
        ts = request.headers.get(HEADER_TS, "")
        nonce = request.headers.get(HEADER_NONCE, "")
        if not sign or not ts or not nonce:
            return JSONResponse(
                {"code": 401, "msg": "gateway_sign_missing"}, status_code=401
            )
        try:
            ts_int = int(ts)
        except ValueError:
            return JSONResponse(
                {"code": 401, "msg": "gateway_sign_invalid_ts"}, status_code=401
            )
        if abs(time.time() - ts_int) > self._ttl:
            return JSONResponse(
                {"code": 401, "msg": "gateway_sign_expired"}, status_code=401
            )

        body = await request.body()
        message = _canonical_message(
            request.method, request.url.path, str(ts_int), nonce, body
        )
        expected = _expected_signature(self._secret, message)
        if not hmac.compare_digest(expected, sign):
            return JSONResponse(
                {"code": 401, "msg": "gateway_sign_invalid"}, status_code=401
            )

        if self._store is not None:
            try:
                first_seen = await self._store.reserve(nonce, self._ttl)
            except Exception as exc:  # pragma: no cover - infra hiccup
                return JSONResponse(
                    {"code": 503, "msg": "gateway_sign_replay_store_unavailable"},
                    status_code=503,
                )
            if not first_seen:
                return JSONResponse(
                    {"code": 409, "msg": "gateway_sign_replay"}, status_code=409
                )

        roles_raw = request.headers.get(HEADER_ROLES, "")
        request.state.user = GatewayUser(
            user_id=request.headers.get(HEADER_USER_ID, "anonymous"),
            username=request.headers.get(HEADER_USERNAME) or None,
            tenant_id=request.headers.get(HEADER_TENANT_ID) or None,
            roles=tuple(r.strip() for r in roles_raw.split(",") if r.strip()),
        )
        return await call_next(request)
