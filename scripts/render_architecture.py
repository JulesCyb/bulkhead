"""Render docs/architecture.svg, the README hero diagram, from the spec below.

The diagram is code, not a binary: change a label here, re-run, commit the SVG. GitHub renders
the file inside an ``<img>``, so it may use only what is embedded — system font stacks, inline
gradients and filters, no external CSS or web fonts.

    uv run python scripts/render_architecture.py            # dark (the variant the README embeds)
    uv run python scripts/render_architecture.py --light    # light, written next to it
"""

from __future__ import annotations

import sys
from pathlib import Path

W, H = 1400, 1080
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

DARK = {
    "bg0": "#0B1220",
    "bg1": "#121C2F",
    "hull": "#111B2E",
    "hull_s": "#25364F",
    "card": "#1A2A45",
    "card_s": "#2F4263",
    "text": "#F3F6FB",
    "text2": "#C4CFE0",
    "muted": "#7C8DA8",
    "line": "#3A4D6B",
    "accent": "#FF5C5C",
    "accent_glow": "#FF5C5C",
    "teal": "#4FD1C5",
    "teal_bg": "#10282F",
    "teal_s": "#237A73",
    "pill": "#152239",
    "pill_s": "#2C3F60",
    "badge_ring": "#0B1220",
    "shadow": "0.35",
}
LIGHT = {
    "bg0": "#F6F8FB",
    "bg1": "#EEF2F7",
    "hull": "#FFFFFF",
    "hull_s": "#D5DDE8",
    "card": "#F3F6FA",
    "card_s": "#CBD5E3",
    "text": "#0E1726",
    "text2": "#33425A",
    "muted": "#7A889D",
    "line": "#B6C2D4",
    "accent": "#E5484D",
    "accent_glow": "#FF7B7B",
    "teal": "#0F8F86",
    "teal_bg": "#E6F7F5",
    "teal_s": "#8FD3CC",
    "pill": "#FFFFFF",
    "pill_s": "#CBD5E3",
    "badge_ring": "#FFFFFF",
    "shadow": "0.08",
}

ARIA = (
    "bulkhead: a multi-tenant AI-agent backend. Three walls the agents cannot cross: each "
    "customer's data is sealed by Postgres Row-Level Security; every change needs a human's "
    "approval; content never leaves its data residency."
)

SUBTITLE = (
    "One backend. Many customers. Any number of agents. Three walls that hold even when an "
    "agent is tricked, the code has a bug, or the config is wrong."
)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Canvas:
    def __init__(self, p: dict[str, str]) -> None:
        self.p = p
        self.out: list[str] = []

    def raw(self, s: str) -> None:
        self.out.append(s)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        color: str,
        weight: int = 400,
        anchor: str = "middle",
        mono: bool = False,
        spacing: float = 0,
    ) -> None:
        family = MONO if mono else SANS
        self.raw(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'letter-spacing="{spacing}">{_esc(s)}</text>'
        )

    def pill(self, x: float, y: float, w: float, h: float, s: str, size: float = 16) -> None:
        p = self.p
        self.raw(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{p["pill"]}" '
            f'stroke="{p["pill_s"]}" stroke-width="1.2"/>'
        )
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size, p["text2"], 500)

    def slab(
        self, x: float, y: float, w: float, h: float, n: str | None = None, badge: str = "top"
    ) -> None:
        p = self.p
        r = min(w, h) / 2
        self.raw(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{p["accent_glow"]}" '
            'opacity="0.55" filter="url(#glow)"/>'
        )
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{p["accent"]}"/>')
        if n:
            cx = x + w / 2 if badge == "top" else x
            cy = y if badge == "top" else y + h / 2
            self.raw(
                f'<circle cx="{cx}" cy="{cy}" r="17" fill="{p["accent"]}" '
                f'stroke="{p["badge_ring"]}" stroke-width="3"/>'
            )
            self.text(cx, cy + 6, n, 17, "#FFFFFF", 700)

    def line(self, x1: float, y1: float, x2: float, y2: float, arrow: bool = False) -> None:
        marker = ' marker-end="url(#arr)"' if arrow else ""
        self.raw(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.p["line"]}" '
            f'stroke-width="1.5"{marker}/>'
        )


def build(p: dict[str, str]) -> str:
    c = _Canvas(p)
    c.raw(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{ARIA}">'
    )
    c.raw(
        "<defs>"
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{p["bg0"]}"/><stop offset="1" stop-color="{p["bg1"]}"/>'
        "</linearGradient>"
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        '<feGaussianBlur stdDeviation="10" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '<filter id="soft" x="-20%" y="-20%" width="140%" height="160%">'
        f'<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" '
        f'flood-opacity="{p["shadow"]}"/></filter>'
        '<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0 1 L9 5 L0 9 z" fill="{p["line"]}"/></marker>'
        "</defs>"
    )
    c.raw(f'<rect width="{W}" height="{H}" rx="28" fill="url(#bg)"/>')

    # Title
    c.raw(
        f'<text x="700" y="74" text-anchor="middle" font-family="{SANS}" font-size="46" '
        f'font-weight="700" fill="{p["text"]}" letter-spacing="-1">bulkhead'
        f'<tspan font-weight="400" fill="{p["accent"]}" letter-spacing="0" dx="18">'
        "secure by design</tspan></text>"
    )
    c.text(700, 110, SUBTITLE, 17, p["muted"])

    # Clients and the front door
    clients = (
        (330, "Your web app", 230),
        (700, "Your mobile app", 230),
        (1070, "Claude Code / any MCP client", 320),
    )
    for cx, label, w in clients:
        c.pill(cx - w / 2, 158, w, 44, label)
        c.line(cx, 202, cx, 246)
    c.raw(
        f'<rect x="200" y="248" width="1000" height="48" rx="24" fill="{p["pill"]}" '
        f'stroke="{p["pill_s"]}" stroke-width="1.2"/>'
    )
    c.text(
        700,
        278,
        "The front door: every request says which customer it belongs to "
        "and carries a signed token",
        16,
        p["text2"],
        500,
    )
    c.text(
        700,
        322,
        "/v1/t/{tenant_id}/…  ·  bearer token verified on every request  ·  "
        "path, token audience and membership must agree",
        12.5,
        p["muted"],
        mono=True,
    )
    c.line(700, 332, 700, 352, arrow=True)

    # The hull
    c.raw(
        f'<rect x="120" y="356" width="1160" height="470" rx="28" fill="{p["hull"]}" '
        f'stroke="{p["hull_s"]}" stroke-width="1.5" filter="url(#soft)"/>'
    )
    c.text(
        160,
        386,
        "ONE DEPLOYMENT · SHARED BY ALL CUSTOMERS",
        12,
        p["muted"],
        600,
        "start",
        spacing=1.6,
    )

    # Wall 1 — sealed compartments (the tenant context is set before any query runs)
    c.text(200, 424, "Wall 1 — sealed compartments", 20, p["text"], 700, "start")
    for x, label in ((200, "Customer A"), (544, "Customer B"), (888, "Customer C")):
        c.raw(
            f'<rect x="{x}" y="454" width="300" height="104" rx="14" fill="{p["teal_bg"]}" '
            f'stroke="{p["teal_s"]}" stroke-width="1.2"/>'
        )
        c.text(x + 150, 498, label, 18, p["teal"], 700)
        c.text(x + 150, 526, "only their data", 15, p["text2"])
    c.slab(516, 438, 12, 138, "1")
    c.slab(860, 438, 12, 138)
    c.text(
        200,
        608,
        "Each customer's data sits in its own compartment. The database enforces the wall — "
        "not the developer, not the agent, not a code review.",
        15.5,
        p["text2"],
        anchor="start",
    )
    c.text(
        200,
        636,
        "Postgres 17  ·  Row-Level Security forced on every table  ·  "
        "the app's DB role has no superuser and cannot bypass RLS",
        11.5,
        p["muted"],
        anchor="start",
        mono=True,
    )

    # Wall 2 — approval (the agents work inside a compartment; a write still needs a person)
    c.raw(
        f'<rect x="160" y="670" width="330" height="122" rx="14" fill="{p["card"]}" '
        f'stroke="{p["card_s"]}" stroke-width="1.2"/>'
    )
    c.text(325, 704, "Your AI agents", 18, p["text"], 700)
    c.text(325, 728, "read, answer, propose changes", 15, p["text2"])
    c.text(325, 750, "never hold a database password", 15, p["text2"])
    c.text(325, 778, "PydanticAI · tools only · run limits", 11.5, p["muted"], mono=True)
    c.line(490, 731, 552, 731, arrow=True)
    c.slab(560, 666, 12, 130, "2")
    c.text(604, 700, "Wall 2 — a human says yes", 20, p["text"], 700, "start")
    c.text(
        604,
        728,
        "Before an agent changes anything, a person approves that exact change.",
        15.5,
        p["text2"],
        anchor="start",
    )
    c.text(604, 752, "There is no “always allow”.", 15.5, p["text2"], anchor="start")
    c.text(
        604,
        780,
        "pending action stored server-side  ·  standing grants for unattended agents  ·  audited",
        11.5,
        p["muted"],
        anchor="start",
        mono=True,
    )

    # Wall 3 — residency
    c.line(700, 826, 700, 854, arrow=True)
    c.slab(300, 862, 800, 12, "3", badge="left")
    c.text(700, 912, "Wall 3 — data stays where it belongs", 20, p["text"], 700)
    c.text(
        700,
        940,
        "An EU customer's text never reaches a US model or a US log. "
        "If the config would allow it, the server refuses to start.",
        15.5,
        p["text2"],
    )
    c.text(
        700,
        964,
        "residency allow-list per customer  ·  routes model, embeddings and traces  ·  "
        "fails closed",
        11.5,
        p["muted"],
        mono=True,
    )
    c.pill(250, 998, 420, 44, "The AI model · a budget per customer", 15)
    c.pill(730, 998, 420, 44, "Monitoring · no message content by default", 15)
    c.text(460, 1064, "LiteLLM gateway", 11.5, p["muted"], mono=True)
    c.text(
        940,
        1064,
        "Langfuse via OpenTelemetry · one sink per residency",
        11.5,
        p["muted"],
        mono=True,
    )
    c.raw("</svg>")
    return "\n".join(c.out)


def main(argv: list[str]) -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    if "--light" in argv:
        target, palette = docs / "architecture-light.svg", LIGHT
    else:
        target, palette = docs / "architecture.svg", DARK
    target.write_text(build(palette), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
