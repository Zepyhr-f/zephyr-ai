from fastapi import FastAPI
from sqlalchemy import text

from app.api.rag import router as rag_router
from app.core.database import engine

app = FastAPI(title="Zephyr AI", version="0.1.0")
app.include_router(rag_router)


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
