"""Persistent campaign memory. JSON-backed, merges new entities into existing ones
with fuzzy name matching. No duplicates for the same entity across sessions."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from difflib import SequenceMatcher


ENTITY_CATEGORIES = [
    "player_characters",
    "npcs",
    "locations",
    "factions",
    "quests",
    "items",
    "events",
    "secrets",
]


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    # Strip common honorifics
    for prefix in ("the ", "sir ", "lord ", "lady ", "captain ", "master ", "miss ", "mr ", "mrs "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _similarity(a: str, b: str) -> float:
    a, b = _normalize_name(a), _normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


class CampaignMemory:
    """Loads and updates a persistent JSON campaign memory file."""

    SCHEMA_VERSION = 1

    def __init__(self, memory_path: Path):
        self.path = memory_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[memory] Failed to load {self.path}: {e}. Starting fresh.")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "sessions": [],  # list of {number, title, date, file}
            **{cat: [] for cat in ENTITY_CATEGORIES},
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def session_count(self) -> int:
        return len(self.data.get("sessions", []))

    def next_session_number(self) -> int:
        return self.session_count + 1

    def record_session(self, session_number: int, title: str, folder: str,
                       summary: str = "") -> None:
        entry = {
            "number": session_number,
            "title": title,
            "date_processed": datetime.utcnow().isoformat(),
            "folder": folder,
            "summary": summary,
        }
        # Replace if reprocessing same number
        self.data["sessions"] = [
            s for s in self.data["sessions"] if s.get("number") != session_number
        ]
        self.data["sessions"].append(entry)
        self.data["sessions"].sort(key=lambda s: s["number"])

    def merge_entities(self, new_entities: Dict[str, List[Dict[str, Any]]],
                       session_number: int) -> Dict[str, int]:
        """Merge extracted entities into the campaign memory.

        Returns a count of {added: N, updated: N} per category for logging.
        """
        stats = {}
        for category in ENTITY_CATEGORIES:
            added = updated = 0
            incoming = new_entities.get(category, []) or []
            existing = self.data.setdefault(category, [])

            for new_item in incoming:
                if not isinstance(new_item, dict):
                    continue
                name = (new_item.get("name") or new_item.get("description") or "").strip()
                if not name:
                    continue

                match_idx = self._find_match(existing, name)
                if match_idx is not None:
                    self._update_entity(existing[match_idx], new_item, session_number)
                    updated += 1
                else:
                    self._add_entity(existing, new_item, session_number)
                    added += 1

            stats[category] = {"added": added, "updated": updated}
        return stats

    def _find_match(self, existing: List[Dict[str, Any]], new_name: str) -> Optional[int]:
        best_idx = None
        best_score = 0.0
        for i, e in enumerate(existing):
            score = _similarity(e.get("name", ""), new_name)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx if best_score >= 0.85 else None

    def _add_entity(self, existing: List[Dict[str, Any]], new_item: Dict[str, Any],
                    session_number: int):
        entry = dict(new_item)
        entry.setdefault("name", new_item.get("name", ""))
        entry["first_session"] = session_number
        entry["latest_session"] = session_number
        entry["session_appearances"] = [session_number]
        entry.setdefault("history", [])
        entry["history"].append({
            "session": session_number,
            "note": new_item.get("description", "") or new_item.get("role", ""),
        })
        existing.append(entry)

    def _update_entity(self, existing_entity: Dict[str, Any], new_item: Dict[str, Any],
                       session_number: int):
        existing_entity["latest_session"] = session_number
        appearances = existing_entity.setdefault("session_appearances", [])
        if session_number not in appearances:
            appearances.append(session_number)

        # Fill missing fields from new data
        for k, v in new_item.items():
            if k == "name":
                continue
            if v and not existing_entity.get(k):
                existing_entity[k] = v

        # Quest status updates: if new data says completed/failed, take it
        if "status" in new_item and new_item["status"] in ("completed", "failed"):
            existing_entity["status"] = new_item["status"]

        history = existing_entity.setdefault("history", [])
        new_note = new_item.get("description", "") or new_item.get("role", "")
        if new_note and not any(h.get("session") == session_number for h in history):
            history.append({"session": session_number, "note": new_note})

    def get_prior_context(self, max_chars: int = 2000) -> str:
        """Produce a short prior-context blurb to feed into the next session's summarizer."""
        if not self.data.get("sessions"):
            return ""
        parts = []
        # Last few session summaries
        recent = self.data["sessions"][-3:]
        if recent:
            parts.append("Recent sessions:")
            for s in recent:
                parts.append(f"- Session {s['number']}: {s['title']}")
                if s.get("summary"):
                    parts.append(f"  {s['summary'][:300]}")

        # Active quests
        active = [q for q in self.data.get("quests", []) if q.get("status", "active") == "active"]
        if active:
            parts.append("\nActive quests:")
            for q in active[:10]:
                parts.append(f"- {q.get('name', '?')}: {q.get('description', '')[:150]}")

        # Recurring NPCs
        npcs = sorted(
            self.data.get("npcs", []),
            key=lambda n: len(n.get("session_appearances", [])),
            reverse=True,
        )[:8]
        if npcs:
            parts.append("\nRecurring NPCs:")
            for n in npcs:
                parts.append(f"- {n.get('name', '?')}: {n.get('description', '')[:120]}")

        context = "\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return context
