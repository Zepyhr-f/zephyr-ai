from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.diagnostics import diagnostics, provider_status
from app.core.version import SERVICE_NAME, SERVICE_VERSION

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/diagnostics")
async def admin_diagnostics(session: AsyncSession = Depends(get_db_session)) -> dict:
    return await diagnostics(session, get_settings())


@router.get("/providers")
async def providers() -> dict:
    return provider_status(get_settings())


@router.get("/version")
async def version() -> dict[str, str]:
    return {"service": SERVICE_NAME, "version": SERVICE_VERSION}
