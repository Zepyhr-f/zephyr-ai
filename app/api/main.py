from fastapi import FastAPI
from sqlalchemy import text

from app.api.admin import router as admin_router
from app.api.conversations import router as conversations_router
from app.api.feedback import router as feedback_router
from app.api.knowledge import router as knowledge_router
from app.api.rag import router as rag_router
from app.core.database import engine
from app.core.gateway_sign import GatewaySignMiddleware
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="Zephyr AI", version="0.1.0")
app.add_middleware(GatewaySignMiddleware)
app.include_router(rag_router)
app.include_router(knowledge_router)
app.include_router(conversations_router)
app.include_router(feedback_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        database = "ok"
    except Exception:
        database = "unavailable"

    status = "ok" if database == "ok" else "degraded"
    return {
        "status": status,
        "service": "zephyr-ai",
        "database": database,
    }
