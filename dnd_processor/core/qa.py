"""Campaign Q&A. Two retrieval modes:

1. Structured-memory mode (default): builds context from campaign_memory.json
   (sessions, NPCs, locations, quests, etc.) and feeds it to the LLM. Best for
   "who/what/when" questions about tracked entities.

2. Semantic mode: uses the vector store to find the most relevant transcript
   chunks for the question, then asks the LLM. Best for "what happened when"
   questions where the relevant detail is in dialogue, not in the entity list.

Both modes fall back to keyword search if no LLM is configured.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from prompts.templates import QA_PROMPT


def _flatten_memory_for_context(memory_data: Dict[str, Any], max_chars: int = 8000) -> str:
    parts = []
    parts.append("=== SESSIONS ===")
    for s in memory_data.get("sessions", []):
        parts.append(f"Session {s.get('number')}: {s.get('title', '')}")
        if s.get("summary"):
            parts.append(s["summary"])
        parts.append("")

    for cat_title, key in [
        ("NPCs", "npcs"),
        ("Locations", "locations"),
        ("Quests", "quests"),
        ("Items", "items"),
        ("Factions", "factions"),
    ]:
        entries = memory_data.get(key, [])
        if not entries:
            continue
        parts.append(f"=== {cat_title} ===")
        for e in entries:
            name = e.get("name", "?")
            desc = e.get("description", "")
            first = e.get("first_session", "?")
            latest = e.get("latest_session", "?")
            status = f" [{e['status']}]" if e.get("status") else ""
            parts.append(f"- {name}{status} (first: S{first}, last: S{latest}): {desc}")
        parts.append("")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"
    return text


class CampaignQA:
    def __init__(self, llm, memory_data: Dict[str, Any], vector_store=None):
        self.llm = llm
        self.memory = memory_data
        self.vector_store = vector_store

    def ask(self, question: str, mode: str = "auto") -> str:
        """mode: 'auto', 'structured', or 'semantic'."""
        if mode == "auto":
            mode = self._choose_mode(question)

        if mode == "semantic" and self.vector_store and self.vector_store.available:
            return self._ask_semantic(question)
        return self._ask_structured(question)

    def _choose_mode(self, question: str) -> str:
        """Pick semantic mode for 'what happened' questions, structured for 'when/who/which'."""
        q = question.lower()
        semantic_triggers = [
            "what happened", "what did", "describe", "tell me about", "what was said",
            "how did", "explain", "what occurred",
        ]
        if any(t in q for t in semantic_triggers):
            return "semantic"
        return "structured"

    # ---- Structured mode ----
    def _ask_structured(self, question: str) -> str:
        if self.llm.is_real_llm:
            context = _flatten_memory_for_context(self.memory)
            prompt = QA_PROMPT.format(context=context, question=question)
            try:
                return self.llm.complete(prompt, max_tokens=600).strip()
            except Exception as e:
                return f"(LLM error: {e})\n\n{self._keyword_search(question)}"
        return self._keyword_search(question)

    # ---- Semantic mode ----
    def _ask_semantic(self, question: str) -> str:
        hits = self.vector_store.search(question, top_k=5)
        if not hits:
            return self._ask_structured(question)

        context_parts = []
        for h in hits:
            context_parts.append(
                f"[Session {h['session_number']}: {h['session_title']}] "
                f"(relevance {h['score']:.2f})\n{h['text']}\n"
            )
        # Also append a short structured-memory blurb so the model has entity context
        memory_blurb = _flatten_memory_for_context(self.memory, max_chars=2000)
        context = "RELEVANT TRANSCRIPT EXCERPTS:\n\n" + "\n---\n".join(context_parts)
        context += f"\n\nCAMPAIGN MEMORY (for entity context):\n{memory_blurb}"

        if self.llm.is_real_llm:
            prompt = QA_PROMPT.format(context=context, question=question)
            try:
                answer = self.llm.complete(prompt, max_tokens=700).strip()
                # Append source sessions for traceability
                sessions_used = sorted({h["session_number"] for h in hits if h.get("session_number")})
                if sessions_used:
                    answer += f"\n\n_Sources: Session {', '.join(f'S{n}' for n in sessions_used)}_"
                return answer
            except Exception as e:
                return f"(LLM error: {e})\n\nTop matching excerpts:\n" + "\n".join(
                    f"[S{h['session_number']}] {h['text'][:300]}..." for h in hits[:3]
                )
        # No LLM: just show the matched excerpts
        return "Top matching transcript excerpts:\n\n" + "\n\n".join(
            f"[Session {h['session_number']}: {h['session_title']}]\n{h['text'][:500]}..."
            for h in hits
        )

    # ---- Keyword fallback ----
    def _keyword_search(self, question: str) -> str:
        keywords = [w.lower() for w in re.findall(r"\w{3,}", question)]
        if not keywords:
            return "No results."
        hits = []
        for s in self.memory.get("sessions", []):
            blob = f"{s.get('title', '')} {s.get('summary', '')}".lower()
            if any(k in blob for k in keywords):
                hits.append(f"[Session {s.get('number')}] {s.get('title')}: {s.get('summary', '')[:200]}")
        for cat in ("npcs", "locations", "quests", "items", "factions"):
            for e in self.memory.get(cat, []):
                blob = f"{e.get('name', '')} {e.get('description', '')}".lower()
                if any(k in blob for k in keywords):
                    hits.append(
                        f"[{cat[:-1].title()}] {e.get('name')}: {e.get('description', '')[:200]} "
                        f"(first: S{e.get('first_session', '?')}, last: S{e.get('latest_session', '?')})"
                    )
        if not hits:
            return "I don't have that information in the campaign notes."
        return "\n".join(hits[:15])
