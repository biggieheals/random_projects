"""Session note generation. Supports long transcripts via map-then-merge."""
from __future__ import annotations
from typing import Callable, Optional, List
from prompts.templates import SESSION_NOTES_PROMPT


def _split_words(text: str, chunk_words: int, overlap: int):
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    out = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        out.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return out


class Summarizer:
    def __init__(self, llm, config, logger: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.config = config
        self.log = logger or (lambda m: None)

    def generate_notes(
        self,
        cleaned_transcript: str,
        prior_context: str = "",
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        chunks = _split_words(
            cleaned_transcript,
            self.config.summary_chunk_words,
            self.config.summary_chunk_overlap,
        )

        context_block = ""
        if prior_context:
            context_block = (
                "PRIOR CAMPAIGN CONTEXT (for continuity only — do not repeat as events of this session):\n"
                f"{prior_context}\n"
            )

        if len(chunks) == 1:
            if progress:
                progress(0.2, "Generating session notes")
            prompt = SESSION_NOTES_PROMPT.format(
                transcript=chunks[0], prior_context=context_block
            )
            notes = self.llm.complete(prompt)
            if progress:
                progress(1.0, "Notes complete")
            return notes.strip()

        # Map: partial notes per chunk, then merge.
        self.log(f"[notes] Long transcript: {len(chunks)} chunks. Map-merge mode.")
        partials = []
        for i, chunk in enumerate(chunks):
            if progress:
                progress(i / (len(chunks) + 1), f"Notes chunk {i+1}/{len(chunks)}")
            prompt = SESSION_NOTES_PROMPT.format(
                transcript=chunk,
                prior_context=context_block + (
                    "\nNOTE: This is one PART of a longer session. Extract what you see; "
                    "later parts will be merged in."
                ),
            )
            partials.append(self.llm.complete(prompt))

        if progress:
            progress(len(chunks) / (len(chunks) + 1), "Merging notes")
        merged = self._merge_partials(partials)
        if progress:
            progress(1.0, "Notes complete")
        return merged

    def _merge_partials(self, partials: List[str]) -> str:
        merge_prompt = (
            "You are merging multiple partial session note sets from different parts of the SAME "
            "D&D session into one coherent set of notes.\n\n"
            "RULES:\n"
            "- Combine without duplicating.\n"
            "- Preserve chronological order.\n"
            "- Do NOT invent new facts.\n"
            "- Keep the same Markdown structure as the inputs.\n\n"
            "PARTIAL NOTES:\n---\n"
            + "\n\n===\n\n".join(partials)
            + "\n---\n\nReturn ONLY the merged Markdown notes."
        )
        try:
            return self.llm.complete(merge_prompt).strip()
        except Exception as e:
            self.log(f"[notes] Merge failed: {e}. Concatenating.")
            return "\n\n---\n\n".join(partials)
