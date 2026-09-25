"""The operator tool (Spec 9, ADR-0010): a command-line entry point (`scripts/operator.py`,
`app.operator.cli`) that connects only as the owner database role (`DATABASE_URL_MIGRATIONS`,
`app/migration_settings.py`) and never the cluster superuser.

This ticket (#68) builds the skeleton every later command shares: audited dispatch
(`app.operator.audit`), a tenant-lookup helper by id or unambiguous name
(`app.operator.lookup`), and the first real command, a read-only tenant listing
(`app.operator.listing`). `create`, `suspend`, and `erase` (later Spec 9 tickets) are built on
top of the same three modules.
"""

from __future__ import annotations
