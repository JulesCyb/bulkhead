"""Shared fixtures: context, fake search, TestModel — no real model call, no DB."""

from __future__ import annotations

import os
import uuid

import pytest
from pydantic_ai.models.test import TestModel

# EMBEDDING_PROVIDER/EMBEDDING_MODEL have no default (app/config.py) — Settings() refuses to
# construct without them. Set process-wide test defaults here, at collection time, so tests that
# don't care about embedding config (most of the suite) don't each have to supply it. Tests that
# exercise the "unset" failure construct Settings(embedding_provider=None, ...) directly, which
# bypasses these env vars entirely.
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")

from app.agents.assistant import AssistantDeps, chat_assistant, one_shot_assistant
from app.context import RequestContext
from app.repositories.documents import DocumentHit


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), roles=frozenset({"member"}))


@pytest.fixture
def calls() -> list[tuple[uuid.UUID, str, int]]:
    return []


@pytest.fixture
def fake_search(calls):
    async def _search(ctx: RequestContext, query: str, limit: int) -> list[DocumentHit]:
        calls.append((ctx.tenant_id, query, limit))
        return [DocumentHit(id=uuid.uuid4(), title="Acme contract", snippet="…", score=0.9)]

    return _search


@pytest.fixture
def deps(ctx, fake_search) -> AssistantDeps:
    return AssistantDeps(ctx=ctx, search=fake_search)


@pytest.fixture
def test_model():
    """TestModel calls every tool once and answers deterministically.

    Overrides both agents (Spec 5 / #36 split) since a test may exercise either the one-shot
    endpoints or /api/chat without knowing in advance which one it will hit.
    """
    tm = TestModel()
    with one_shot_assistant.override(model=tm), chat_assistant.override(model=tm):
        yield tm
