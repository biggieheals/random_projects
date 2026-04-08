"""Transcript cleanup. Splits into chunks and runs them through the LLM cleanup prompt."""
from __future__ import annotations
from typing import Callable, Optional
from prompts.templates import CLEANUP_PROMPT


def _split_into_chunks(text: str, chunk_words: int, overlap_words: int):
    words = text.split()
    if len(words) <= chunk_words:
        yield text
        return
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        yield " ".join(words[start:end])
        if end >= len(words):
            break
        start = end - overlap_words


class Cleaner:
    def __init__(self, llm, config, logger: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.config = config
        self.log = logger or (lambda m: None)

    def clean(self, raw_text: str, progress: Optional[Callable[[float, str], None]] = None) -> str:
        chunks = list(_split_into_chunks(
            raw_text,
            self.config.cleanup_chunk_words,
            self.config.cleanup_chunk_overlap,
        ))
        self.log(f"[clean] Processing {len(chunks)} chunk(s).")
        cleaned_parts = []
        for i, chunk in enumerate(chunks):
            if progress:
                progress(i / len(chunks), f"Cleaning chunk {i+1}/{len(chunks)}")
            try:
                prompt = CLEANUP_PROMPT.format(transcript=chunk)
                result = self.llm.complete(prompt)
                cleaned_parts.append(result.strip() if result else chunk)
            except Exception as e:
                self.log(f"[clean] Chunk {i+1} failed: {e}. Keeping raw chunk.")
                cleaned_parts.append(chunk)
        if progress:
            progress(1.0, "Cleanup complete")
        return "\n\n".join(cleaned_parts)
