"""Markdown wiki export from campaign memory. Obsidian-friendly."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List


def _fmt_sessions(appearances: List[int]) -> str:
    if not appearances:
        return ""
    return ", ".join(f"[[session_{n:03d}]]" for n in sorted(set(appearances)))


def _header(title: str, count: int) -> str:
    return f"# {title}\n\n*{count} entries*\n\n"


def export_wiki(memory_data: Dict[str, Any], wiki_dir: Path) -> List[Path]:
    wiki_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # NPC index
    npcs = memory_data.get("npcs", [])
    lines = [_header("NPC Index", len(npcs))]
    for n in sorted(npcs, key=lambda x: x.get("name", "").lower()):
        name = n.get("name", "Unknown")
        lines.append(f"## {name}\n")
        if n.get("description"):
            lines.append(f"{n['description']}\n")
        if n.get("role"):
            lines.append(f"- **Role:** {n['role']}")
        if n.get("location"):
            lines.append(f"- **Location:** {n['location']}")
        if n.get("status"):
            lines.append(f"- **Status:** {n['status']}")
        if n.get("first_session"):
            lines.append(f"- **First seen:** Session {n['first_session']}")
        if n.get("latest_session"):
            lines.append(f"- **Last seen:** Session {n['latest_session']}")
        appearances = _fmt_sessions(n.get("session_appearances", []))
        if appearances:
            lines.append(f"- **Appearances:** {appearances}")
        lines.append("")
    p = wiki_dir / "npc_index.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    # Locations
    locs = memory_data.get("locations", [])
    lines = [_header("Location Index", len(locs))]
    for l in sorted(locs, key=lambda x: x.get("name", "").lower()):
        lines.append(f"## {l.get('name', 'Unknown')}\n")
        if l.get("description"):
            lines.append(f"{l['description']}\n")
        if l.get("type"):
            lines.append(f"- **Type:** {l['type']}")
        if l.get("first_session"):
            lines.append(f"- **First visited:** Session {l['first_session']}")
        appearances = _fmt_sessions(l.get("session_appearances", []))
        if appearances:
            lines.append(f"- **Appearances:** {appearances}")
        lines.append("")
    p = wiki_dir / "location_index.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    # Quest log
    quests = memory_data.get("quests", [])
    active = [q for q in quests if q.get("status", "active") == "active"]
    other = [q for q in quests if q.get("status", "active") != "active"]
    lines = [_header("Quest Log", len(quests))]
    lines.append("## Active\n")
    for q in active:
        lines.append(f"### {q.get('name', 'Unknown')}")
        if q.get("description"):
            lines.append(q["description"])
        if q.get("first_session"):
            lines.append(f"- Started: Session {q['first_session']}")
        lines.append("")
    lines.append("## Resolved / Other\n")
    for q in other:
        lines.append(f"### {q.get('name', 'Unknown')} — *{q.get('status', 'unknown')}*")
        if q.get("description"):
            lines.append(q["description"])
        lines.append("")
    p = wiki_dir / "quest_log.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    # Items
    items = memory_data.get("items", [])
    lines = [_header("Item Log", len(items))]
    for i in sorted(items, key=lambda x: x.get("name", "").lower()):
        lines.append(f"## {i.get('name', 'Unknown')}")
        if i.get("description"):
            lines.append(i["description"])
        if i.get("owner"):
            lines.append(f"- **Owner:** {i['owner']}")
        if i.get("type"):
            lines.append(f"- **Type:** {i['type']}")
        if i.get("first_session"):
            lines.append(f"- **Acquired:** Session {i['first_session']}")
        lines.append("")
    p = wiki_dir / "item_log.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    # Factions
    factions = memory_data.get("factions", [])
    lines = [_header("Faction Index", len(factions))]
    for f in sorted(factions, key=lambda x: x.get("name", "").lower()):
        lines.append(f"## {f.get('name', 'Unknown')}")
        if f.get("description"):
            lines.append(f["description"])
        if f.get("disposition"):
            lines.append(f"- **Disposition:** {f['disposition']}")
        lines.append("")
    p = wiki_dir / "faction_index.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    # Session index
    sessions = memory_data.get("sessions", [])
    lines = [_header("Session Index", len(sessions))]
    for s in sessions:
        num = s.get("number", "?")
        lines.append(f"## Session {num}: {s.get('title', 'Untitled')}")
        if s.get("date_processed"):
            lines.append(f"*Processed: {s['date_processed']}*")
        if s.get("summary"):
            lines.append(f"\n{s['summary']}")
        if s.get("folder"):
            lines.append(f"\n[Open notes]({s['folder']})")
        lines.append("")
    p = wiki_dir / "session_index.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)

    return written
