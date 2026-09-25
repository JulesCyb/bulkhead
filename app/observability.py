"""Tracing for agent runs (Langfuse via OpenTelemetry, optional).

Active once LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY are set and `logfire` is installed
(`uv sync --extra observability`). Context attributes (tenant_id, identity_id, request_id) are
attached by the agent call via `metadata` — see app/agents/assistant.py.
"""

from __future__ import annotations

import base64
import logging
import os

from app.config import Settings

log = logging.getLogger(__name__)


def setup_observability(settings: Settings) -> bool:
    if not (
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    ):
        return False
    secret_key = (
        settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else ""
    )
    auth = base64.b64encode(f"{settings.langfuse_public_key}:{secret_key}".encode()).decode()
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_ENDPOINT", f"{settings.langfuse_host.rstrip('/')}/api/public/otel"
    )
    os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {auth}")
    try:
        import logfire
    except ImportError:
        log.warning("Langfuse configured but `logfire` is missing: uv sync --extra observability")
        return False
    logfire.configure(service_name=settings.app_name, send_to_logfire=False)
    logfire.instrument_pydantic_ai()
    return True
