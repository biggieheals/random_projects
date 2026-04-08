"""Persistent speaker voice-print registry for cross-session speaker recognition.

Each campaign has its own speaker_registry.json:
{
  "speakers": {
    "Jess": {
      "embedding": [0.123, -0.456, ...],   # 192-d or 256-d depending on model
      "sessions": [1, 2, 3],
      "sample_count": 4
    },
    ...
  },
  "match_threshold": 0.65
}

On a new session we extract one embedding per detected pyannote speaker, then for each
unknown embedding we compute cosine similarity against every known voice. If the best
match is above threshold we reuse the name; otherwise we leave it as SPEAKER_X for the
user to label in the GUI.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SpeakerRegistry:
    def __init__(self, path: Path, default_threshold: float = 0.65):
        self.path = path
        self.data = self._load(default_threshold)

    def _load(self, default_threshold: float) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"speakers": {}, "match_threshold": default_threshold}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def threshold(self) -> float:
        return float(self.data.get("match_threshold", 0.65))

    @property
    def known_names(self) -> List[str]:
        return list(self.data.get("speakers", {}).keys())

    def match(self, embedding: List[float]) -> Tuple[Optional[str], float]:
        """Return (best_name, similarity). best_name is None if below threshold."""
        speakers = self.data.get("speakers", {})
        if not speakers or not embedding:
            return None, 0.0
        best_name, best_sim = None, -1.0
        for name, info in speakers.items():
            sim = cosine_similarity(embedding, info.get("embedding", []))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= self.threshold:
            return best_name, best_sim
        return None, best_sim

    def add_or_update(self, name: str, embedding: List[float], session_number: int):
        """Add a new known speaker or refine an existing voice print.

        Refinement uses a weighted running average so the embedding adapts over many
        sessions without any single bad sample dominating.
        """
        speakers = self.data.setdefault("speakers", {})
        if name in speakers:
            existing = speakers[name]
            old_emb = existing.get("embedding", [])
            count = existing.get("sample_count", 1)
            if old_emb and len(old_emb) == len(embedding):
                # Running average
                blended = [
                    (o * count + n) / (count + 1)
                    for o, n in zip(old_emb, embedding)
                ]
                existing["embedding"] = blended
            else:
                existing["embedding"] = list(embedding)
            existing["sample_count"] = count + 1
            sessions = existing.setdefault("sessions", [])
            if session_number not in sessions:
                sessions.append(session_number)
        else:
            speakers[name] = {
                "embedding": list(embedding),
                "sessions": [session_number],
                "sample_count": 1,
            }

    def rename(self, old: str, new: str):
        speakers = self.data.get("speakers", {})
        if old in speakers and new and new != old:
            speakers[new] = speakers.pop(old)

    def remove(self, name: str):
        self.data.get("speakers", {}).pop(name, None)
