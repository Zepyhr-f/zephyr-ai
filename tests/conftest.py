import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

engine = (
    create_async_engine(TEST_DATABASE_URL, echo=False, future=True, poolclass=NullPool)
    if TEST_DATABASE_URL
    else None
)
AsyncTestSession = (
    sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) if engine else None
)


@pytest_asyncio.fixture(scope="session")
async def setup_database() -> AsyncGenerator[None, None]:
    if engine is None:
        pytest.skip("TEST_DATABASE_URL is not configured; skipping database integration tests")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(setup_database: None) -> AsyncGenerator[AsyncSession, None]:
    if AsyncTestSession is None:
        pytest.skip("TEST_DATABASE_URL is not configured; skipping database integration tests")
    async with AsyncTestSession() as session:
        yield session
        await session.rollback()
