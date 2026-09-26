"""The single-path rule (spec A5 / #113, #114, spec #95's Testing Decisions): no SQL string
outside the control repository mentions the `control` schema. Parses every module under `app/`
and `scripts/` with `ast` (never a text grep, which would also match a comment or a docstring
quoting the same schema-qualified name -- see `app/db/session.py`'s own module docstring,
`app/tenant_record.py`'s, etc.) and inspects only the string literals actually passed to a call
named `text` (SQLAlchemy's `text()`, this codebase's only way to issue a raw statement).

No exemption for anything #113/#114 touched: `app/db/session.py` (the session router),
`app/db/guard.py` (the guard), `scripts/migrate.py` (the migration runner), every operator command
(`create.py`, `erase.py`, `suspend.py`, `listing.py`, `lookup.py`, `audit.py`, `dedicated_db.py`),
and `app/gateway_provisioning.py` all read and write the control schema through
`ControlRepository` now and carry no SQL string of their own that names it.

One exemption remains, exactly as spec A5 documents:

- `app/repositories/control.py` -- the repository itself, the one module allowed to.

One more, pre-existing and unrelated to spec A5: `app/repositories/agent_identities.py` calls
`control.create_agent_identity()` (ADR-0005, Spec 6 / #46) -- a different repository, for a
different control-plane fact (minting an agent identity), that predates this spec and is not one
of the "five hand-written SQL strings in five modules" spec #95's problem statement names.
Tightening it is its own concern, not something #113/#114 claims to fix; it is named here, rather
than the grep silently passing, so removing this exemption later is a one-line, intentional
decision.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CONTROL_SCHEMA = re.compile(r"\bcontrol\.[A-Za-z_]")

_REPO_ROOT = Path(__file__).resolve().parent.parent

_EXEMPT_FILES = {
    _REPO_ROOT / "app" / "repositories" / "control.py",
    _REPO_ROOT / "app" / "repositories" / "agent_identities.py",
}
_EXEMPT_DIRS: set[Path] = set()


def _is_exempt(path: Path) -> bool:
    if path in _EXEMPT_FILES:
        return True
    return any(directory in path.parents for directory in _EXEMPT_DIRS)


def _text_call_strings(tree: ast.AST) -> list[str]:
    """Every string literal that appears anywhere inside a call named `text(...)` in `tree` --
    walking the whole call, not just its first argument, so a statement built from more than one
    string constant (implicit concatenation, or a helper that passes a second positional/keyword
    argument) is still caught."""
    found: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "text":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        found.append(sub.value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return found


def _offending_files() -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for base in ("app", "scripts"):
        for path in sorted((_REPO_ROOT / base).rglob("*.py")):
            if _is_exempt(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            matches = [s for s in _text_call_strings(tree) if _CONTROL_SCHEMA.search(s)]
            if matches:
                offenders[str(path.relative_to(_REPO_ROOT))] = matches
    return offenders


def test_no_sql_string_outside_the_control_repository_mentions_the_control_schema():
    offenders = _offending_files()
    assert offenders == {}, (
        "the following file(s) issue SQL naming the `control` schema outside "
        "app/repositories/control.py and app/repositories/agent_identities.py (see this test's "
        "own module docstring for why that second one is exempt): " + repr(offenders)
    )


def test_the_exemption_list_itself_still_points_at_real_files():
    """A typo in `_EXEMPT_FILES`/`_EXEMPT_DIRS` would silently widen the rule above (a path that
    no longer exists exempts nothing, but a *misspelled* one exempts nothing either while looking
    like it does) -- this catches the second case by asserting every exempt path is real."""
    for path in _EXEMPT_FILES:
        assert path.is_file(), path
    for directory in _EXEMPT_DIRS:
        assert directory.is_dir(), directory
