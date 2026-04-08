"""Narrative campaign journal rewrite."""
from __future__ import annotations
from typing import Optional, Callable
from prompts.templates import JOURNAL_PROMPT


class Journalist:
    def __init__(self, llm, logger: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.log = logger or (lambda m: None)

    def rewrite(self, session_notes: str) -> str:
        try:
            return self.llm.complete(JOURNAL_PROMPT.format(notes=session_notes)).strip()
        except Exception as e:
            self.log(f"[journal] Rewrite failed: {e}")
            return "# Campaign Journal\n\n*(Journal generation failed. See session_notes.md.)*"
