"""Pool sizing is explicit, named configuration, not driver defaults (Spec 7 / #55).

Constructing the async engine never opens a connection, so this runs without a real database.
"""

from __future__ import annotations

from app import config
from app.config import Settings
from app.db import session as db_session


def test_engine_uses_explicit_pool_settings(monkeypatch):
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        db_pool_size=7,
        db_max_overflow=3,
        db_pool_timeout=11,
        db_pool_recycle=222,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(db_session, "get_settings", lambda: settings)
    db_session._engine = None
    db_session._session_factory = None
    try:
        engine = db_session.get_engine()
        pool = engine.pool
        assert pool.size() == 7
        assert pool._max_overflow == 3
        assert pool._timeout == 11
        assert pool._recycle == 222
    finally:
        db_session._engine = None
        db_session._session_factory = None


def test_pool_settings_are_named_configuration_not_defaults():
    defaults = Settings(database_url="postgresql+asyncpg://app:app@localhost:5432/app")
    assert defaults.db_pool_size > 0
    assert defaults.db_max_overflow >= 0
    assert defaults.db_pool_timeout > 0
    assert defaults.db_pool_recycle > 0
    assert defaults.db_statement_timeout_ms > 0
