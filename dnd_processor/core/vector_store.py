"""Optional vector search over session transcripts using chromadb.

If chromadb isn't installed, VectorStore.available is False and the rest of the app
falls back to keyword search. We use chromadb's built-in default embedder
(sentence-transformers/all-MiniLM-L6-v2) so there are no API calls or extra deps
beyond chromadb itself.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable


def _split_into_chunks(text: str, words_per_chunk: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text] if words else []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


class VectorStore:
    """Per-campaign vector store. Lazy: only loads chromadb on first use."""

    def __init__(self, campaign_dir: Path, logger: Optional[Callable[[str], None]] = None):
        self.campaign_dir = campaign_dir
        self.store_dir = campaign_dir / "vector_store"
        self.log = logger or (lambda m: None)
        self._client = None
        self._collection = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import chromadb  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            self.log("[vector] chromadb not installed; vector search disabled. "
                     "(pip install chromadb)")
        return self._available

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if not self.available:
            return None
        import chromadb
        self.store_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(self.store_dir))
            self._collection = self._client.get_or_create_collection(
                name="sessions",
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception as e:
            self.log(f"[vector] Failed to open chromadb collection: {e}")
            self._available = False
            return None

    def index_session(
        self,
        session_number: int,
        session_title: str,
        cleaned_text: str,
    ) -> int:
        """Index (or re-index) a session's cleaned transcript. Returns chunks added."""
        col = self._get_collection()
        if col is None:
            return 0

        # Remove any prior chunks for this session number (idempotent reprocess)
        try:
            col.delete(where={"session_number": session_number})
        except Exception:
            pass

        chunks = _split_into_chunks(cleaned_text)
        if not chunks:
            return 0

        ids = [f"s{session_number:04d}_c{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {
                "session_number": session_number,
                "session_title": session_title,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        try:
            col.add(ids=ids, documents=chunks, metadatas=metadatas)
            self.log(f"[vector] Indexed {len(chunks)} chunk(s) for session {session_number}.")
            return len(chunks)
        except Exception as e:
            self.log(f"[vector] Indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        col = self._get_collection()
        if col is None:
            return []
        try:
            results = col.query(query_texts=[query], n_results=top_k)
        except Exception as e:
            self.log(f"[vector] Query failed: {e}")
            return []

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "text": doc,
                "session_number": meta.get("session_number"),
                "session_title": meta.get("session_title"),
                "score": 1.0 - float(dist),  # cosine distance -> similarity-ish
            })
        return out

    def stats(self) -> Dict[str, Any]:
        col = self._get_collection()
        if col is None:
            return {"available": False, "count": 0}
        try:
            return {"available": True, "count": col.count()}
        except Exception:
            return {"available": True, "count": 0}
