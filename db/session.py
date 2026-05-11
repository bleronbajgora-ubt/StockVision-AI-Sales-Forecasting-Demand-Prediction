"""
=============================================================================
 app/db/session.py — Async Database Engine & Session Factory
=============================================================================

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_engine:          Optional[AsyncEngine]       = None
_session_factory: Optional[async_sessionmaker] = None


def _build_engine() -> AsyncEngine:
    """
    Create the SQLAlchemy async engine from the active DATABASE_URL.

    WHY called lazily:
      Tests set DATABASE_URL=sqlite+aiosqlite:///:memory: via environment
      variable before the first import. Building the engine at import time
      would attempt to load asyncpg even in the test environment where it
      is unavailable, crashing before a single test can run.

    WHY pool_pre_ping=True:
      Verifies each pooled connection with a cheap SELECT 1 before use.
      Prevents 'server closed the connection unexpectedly' errors after
      PostgreSQL restarts or idle-connection timeouts.
    """
    from app.core.settings import settings

    url = settings.DATABASE_URL

    # SQLite (tests) does not support pool_size / max_overflow
    is_sqlite = url.startswith("sqlite")

    kwargs = dict(echo=settings.is_development, pool_pre_ping=not is_sqlite)
    if not is_sqlite:
        kwargs.update(
            pool_size    = settings.DB_POOL_SIZE,
            max_overflow = settings.DB_MAX_OVERFLOW,
            pool_timeout = settings.DB_POOL_TIMEOUT,
        )

    return create_async_engine(url, **kwargs)


def get_engine() -> AsyncEngine:
    """Return the module-level singleton engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Return the module-level singleton session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind         = get_engine(),
            class_       = AsyncSession,
            expire_on_commit = False,
            autocommit   = False,
            autoflush    = False,
        )
    return _session_factory


def reset_engine() -> None:
    """Reset singletons — used by tests to swap DATABASE_URL between runs."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None


# ── FastAPI Dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield one AsyncSession per HTTP request.

    Commits on clean exit, rolls back on any exception, always closes.
    FastAPI calls this via Depends(get_db) in every endpoint that needs the DB.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Non-request context manager ───────────────────────────────────────────────
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Use outside a request (CLI scripts, background tasks, tests).

        async with get_db_context() as db:
            results = await db.execute(...)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()