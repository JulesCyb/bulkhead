"""`app.db.lifecycle.owner_engine` (spec A5 / #113): build, yield, dispose -- no real database
needed. `create_async_engine` itself is faked (`AsyncEngine` has no settable `dispose` attribute
to spy on directly) so these tests never attempt a real connection."""

from __future__ import annotations

import pytest

import app.db.lifecycle as lifecycle_module
from app.db.lifecycle import owner_engine


class _FakeEngine:
    def __init__(self, dsn: str, **kwargs: object) -> None:
        self.dsn = dsn
        self.kwargs = kwargs
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture
def fake_create_async_engine(monkeypatch):
    built: list[_FakeEngine] = []

    def _fake(dsn: str, **kwargs: object) -> _FakeEngine:
        engine = _FakeEngine(dsn, **kwargs)
        built.append(engine)
        return engine

    monkeypatch.setattr(lifecycle_module, "create_async_engine", _fake)
    return built


async def test_owner_engine_yields_the_built_engine_and_disposes_it_on_clean_exit(
    fake_create_async_engine,
):
    async with owner_engine("postgresql+asyncpg://user@localhost/db") as engine:
        assert engine is fake_create_async_engine[0]
        assert engine.disposed is False
    assert engine.disposed is True


async def test_owner_engine_disposes_even_when_the_caller_raises(fake_create_async_engine):
    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with owner_engine("postgresql+asyncpg://user@localhost/db"):
            raise _Boom("simulated failure")

    assert fake_create_async_engine[0].disposed is True


async def test_owner_engine_forwards_kwargs_to_create_async_engine(fake_create_async_engine):
    async with owner_engine(
        "postgresql+asyncpg://user@localhost/db", isolation_level="AUTOCOMMIT"
    ) as engine:
        assert engine.kwargs == {"isolation_level": "AUTOCOMMIT"}
        assert engine.dsn == "postgresql+asyncpg://user@localhost/db"
