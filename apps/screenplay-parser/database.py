"""
Postgres async engine and session for screenplay-parser.

Uses SQLAlchemy 2.x async with asyncpg. Set DATABASE_URL in the environment, e.g.:
  postgresql+asyncpg://user:password@localhost:5432/dbname
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Default for local dev when infra postgres is used (align with scripts/start.sh: dev/dev/dev)
DEFAULT_URL = "postgresql+asyncpg://dev:dev@localhost:5432/dev"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_URL)

# Ensure async driver for asyncpg (replace postgresql:// with postgresql+asyncpg://)
if DATABASE_URL.startswith("postgresql://") and "asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "0").lower() in ("1", "true", "yes"),
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session; commits on exit, rolls back on exception."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create pgvector extension and all tables. Safe to call on startup."""
    from sqlalchemy import text

    import models  # noqa: F401 - register models with Base

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Add columns if missing (existing DBs)
    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE screenplays ADD COLUMN IF NOT EXISTS content JSONB")
        )
        await conn.execute(
            text(
                "ALTER TABLE screenplays ADD COLUMN IF NOT EXISTS is_deleted boolean NOT NULL DEFAULT false"
            )
        )
