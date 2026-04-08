"""LLM abstraction supporting OpenAI, Anthropic, Ollama, and a heuristic fallback.

The OpenAI backend also works with any OpenAI-compatible endpoint:
- OpenRouter: base_url=https://openrouter.ai/api/v1
- LM Studio:  base_url=http://localhost:1234/v1
- vLLM:       base_url=http://localhost:8000/v1
- Groq:       base_url=https://api.groq.com/openai/v1
- Together:   base_url=https://api.together.xyz/v1
"""
from __future__ import annotations
import json
import re
from typing import Optional, Callable


class LLMError(Exception):
    pass


SUPPORTED_BACKENDS = ["fallback", "openai", "anthropic", "ollama"]


class LLM:
    def __init__(self, config, logger: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log = logger or (lambda msg: None)
        self.backend = config.llm_backend
        self._client = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == "openai":
            try:
                from openai import OpenAI
                if not self.config.openai_api_key:
                    self.log("[llm] No OpenAI key set; falling back to heuristic mode.")
                    self.backend = "fallback"
                    return
                self._client = OpenAI(
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                )
                self.log(f"[llm] OpenAI-compatible backend ready "
                         f"(model={self.config.openai_model}, url={self.config.openai_base_url}).")
            except ImportError:
                self.log("[llm] openai package not installed; falling back. (pip install openai)")
                self.backend = "fallback"

        elif self.backend == "anthropic":
            try:
                import anthropic
                if not self.config.anthropic_api_key:
                    self.log("[llm] No Anthropic key set; falling back.")
                    self.backend = "fallback"
                    return
                self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                self.log(f"[llm] Anthropic backend ready (model={self.config.anthropic_model}).")
            except ImportError:
                self.log("[llm] anthropic package not installed; falling back. (pip install anthropic)")
                self.backend = "fallback"

        elif self.backend == "ollama":
            try:
                import requests  # noqa: F401
                self.log(f"[llm] Ollama backend ready "
                         f"(model={self.config.ollama_model}, url={self.config.ollama_base_url}).")
            except ImportError:
                self.log("[llm] requests not installed; falling back.")
                self.backend = "fallback"
        else:
            self.log("[llm] Using fallback heuristic mode (no LLM).")

    @property
    def is_real_llm(self) -> bool:
        return self.backend in ("openai", "anthropic", "ollama")

    def complete(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        max_tokens = max_tokens or self.config.llm_max_tokens
        temperature = temperature if temperature is not None else self.config.llm_temperature

        if self.backend == "openai":
            return self._complete_openai(prompt, max_tokens, temperature)
        if self.backend == "anthropic":
            return self._complete_anthropic(prompt, max_tokens, temperature)
        if self.backend == "ollama":
            return self._complete_ollama(prompt, max_tokens, temperature)
        return self._complete_fallback(prompt)

    def _complete_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            self.log(f"[llm] OpenAI error: {e}")
            raise LLMError(str(e))

    def _complete_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            resp = self._client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Concatenate text blocks
            parts = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "".join(parts)
        except Exception as e:
            self.log(f"[llm] Anthropic error: {e}")
            raise LLMError(str(e))

    def _complete_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import requests
        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            self.log(f"[llm] Ollama error: {e}")
            raise LLMError(str(e))

    # ---- Fallback heuristic mode ----
    def _complete_fallback(self, prompt: str) -> str:
        if "cleaning up a raw transcript" in prompt:
            return self._fallback_cleanup(prompt)
        if "generate structured session notes" in prompt.lower() or "session title" in prompt.lower():
            return self._fallback_session_notes(prompt)
        if "Extract structured campaign entities" in prompt:
            return self._fallback_entities(prompt)
        if "Rewrite these D&D session notes" in prompt:
            return self._fallback_journal(prompt)
        if "resolving whether two campaign entities" in prompt:
            return "NO"
        if prompt.strip().startswith("You are answering a question"):
            return "I don't have that information in the campaign notes."
        return ""

    def _extract_between(self, prompt: str, marker: str = "---") -> str:
        parts = prompt.split(marker)
        if len(parts) >= 3:
            return parts[1].strip()
        return ""

    def _fallback_cleanup(self, prompt: str) -> str:
        text = self._extract_between(prompt)
        fillers = r"\b(um+|uh+|er+|ah+|like|you know|i mean|sort of|kind of)\b,?\s*"
        text = re.sub(fillers, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fallback_session_notes(self, prompt: str) -> str:
        transcript = self._extract_between(prompt)
        lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
        preview = " ".join(lines)[:600]
        return f"""# Session (Auto-generated)

## Short Summary
{preview}...

## Scene-by-Scene Breakdown
- Automatic breakdown not available in fallback mode. See cleaned_transcript.txt for full content.

## Key Events
- Fallback mode: key events not extracted. Configure an LLM backend for full extraction.

## Character Highlights
None noted

## Important NPCs
None noted

## Important Locations
None noted

## Loot / Rewards / Items
None noted

## Quests / Hooks / Unresolved Threads
None noted

## Consequences for Future Sessions
None noted
"""

    def _fallback_entities(self, prompt: str) -> str:
        text = self._extract_between(prompt)
        candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
        stop = {"Session", "None", "Summary", "Scene", "Key", "Events", "Character",
                "Important", "Loot", "Rewards", "Items", "Quests", "Hooks", "Unresolved",
                "Consequences", "Future", "Sessions", "The", "A", "An", "This", "That"}
        uniq = []
        seen = set()
        for c in candidates:
            if c in stop or c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        npcs = [{"name": n, "description": "", "role": "", "status": "", "location": ""}
                for n in uniq[:10]]
        payload = {
            "player_characters": [],
            "npcs": npcs,
            "locations": [],
            "factions": [],
            "quests": [],
            "items": [],
            "events": [],
            "secrets": [],
        }
        return json.dumps(payload, indent=2)

    def _fallback_journal(self, prompt: str) -> str:
        notes = self._extract_between(prompt)
        first_lines = "\n".join(notes.splitlines()[:20])
        return f"# Campaign Journal (Auto-generated)\n\n{first_lines}\n\n*(Journal rewrite unavailable in fallback mode.)*"
