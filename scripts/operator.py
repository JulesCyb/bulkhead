"""Operator tool entry point (Spec 9 / #68): connects only as the owner database role, never
the cluster superuser. See `app/operator/` for the implementation (audited dispatch, the
tenant-lookup helper, and each command).

    uv run python scripts/operator.py list
    uv run python scripts/operator.py suspend <tenant-id-or-name>
    uv run python scripts/operator.py unsuspend <tenant-id-or-name>
"""

from __future__ import annotations

from app.operator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
