"""Entity extraction from session notes."""
from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional, Callable
from prompts.templates import ENTITY_EXTRACTION_PROMPT


EMPTY_ENTITIES: Dict[str, list] = {
    "player_characters": [],
    "npcs": [],
    "locations": [],
    "factions": [],
    "quests": [],
    "items": [],
    "events": [],
    "secrets": [],
}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


class Extractor:
    def __init__(self, llm, logger: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.log = logger or (lambda m: None)

    def extract(self, session_notes: str) -> Dict[str, Any]:
        prompt = ENTITY_EXTRACTION_PROMPT.format(notes=session_notes)
        try:
            raw = self.llm.complete(prompt)
        except Exception as e:
            self.log(f"[extract] LLM call failed: {e}")
            return dict(EMPTY_ENTITIES)

        raw = _strip_code_fence(raw)
        # Try to locate the JSON object if the model wrapped it in prose
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self.log(f"[extract] JSON parse failed: {e}. Returning empty structure.")
            return dict(EMPTY_ENTITIES)

        # Normalise: ensure all keys exist
        result = dict(EMPTY_ENTITIES)
        for key in result:
            val = data.get(key)
            if isinstance(val, list):
                result[key] = val
        return result
