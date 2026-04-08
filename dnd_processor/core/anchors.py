"""Timestamp anchoring for session notes.

The session notes are generated from cleaned text without timestamps. To make events
clickable/jumpable, we match each Key Event bullet against the original timestamped
raw transcript via fuzzy text overlap, then write the best timestamp into the notes.

Strategy:
- Parse timestamped lines from raw_transcript.txt -> list of (start_seconds, file, text)
- For each event bullet, score every transcript window of similar length using
  word-overlap. Take the highest-scoring window's timestamp.
- Threshold scores to avoid bogus anchors (events with no good match get no anchor).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


_TIMESTAMP_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(?:[A-Za-z0-9_]+:\s*)?(.*)$")
_FILE_HEADER_RE = re.compile(r"^---\s*(.+?)\s*---$")


def parse_timestamped_transcript(raw_text: str) -> List[Dict[str, Any]]:
    """Return [{seconds, file, text}, ...] from raw_transcript.txt content."""
    out: List[Dict[str, Any]] = []
    current_file: Optional[str] = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _FILE_HEADER_RE.match(line)
        if m:
            current_file = m.group(1)
            continue
        m = _TIMESTAMP_RE.match(line)
        if m:
            h, mm, s, txt = m.groups()
            seconds = int(h) * 3600 + int(mm) * 60 + int(s)
            out.append({"seconds": seconds, "file": current_file, "text": txt.strip()})
    return out


def _format_timestamp(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalize_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]


_COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "his", "had", "has", "have", "they", "this", "that",
    "with", "from", "your", "their", "them", "then", "than", "into", "what",
    "when", "where", "who", "why", "how", "will", "would", "could", "should",
    "there", "here", "some", "these", "those", "say", "said", "says", "get", "got",
    "make", "made", "see", "saw", "seen", "come", "came", "take", "took", "give",
    "gave", "find", "found", "know", "knew", "think", "thought", "look", "looked",
    "want", "tell", "told", "ask", "asked", "feel", "felt", "try", "tried",
    "leave", "left", "call", "called", "good", "great", "right", "first",
    "last", "long", "much", "many", "more", "most", "very", "well", "also", "just",
    "even", "still", "back", "down", "over", "after", "before", "again", "around",
    "really", "okay", "yeah", "yes", "now", "appeared",
}


def _score_overlap(query_words: set, candidate_words: List[str]) -> float:
    """Score with rarity weighting: rare overlapping words count more than common ones."""
    if not query_words or not candidate_words:
        return 0.0
    cand_set = set(candidate_words)
    overlap = query_words & cand_set
    if not overlap:
        return 0.0
    rare_overlap = sum(1 for w in overlap if w not in _COMMON_WORDS)
    rare_query = sum(1 for w in query_words if w not in _COMMON_WORDS)
    common_overlap = len(overlap) - rare_overlap
    if rare_query == 0:
        return len(overlap) / max(len(query_words), 1)
    return (rare_overlap * 1.0 + common_overlap * 0.2) / max(rare_query, 1)


def find_anchor_for_text(
    target_text: str,
    transcript_lines: List[Dict[str, Any]],
    min_score: float = 1.0,
    window_size: int = 3,
) -> Optional[Dict[str, Any]]:
    """Find the best timestamp anchor by scoring each transcript line. The line's own
    text counts double; the window of neighbors gives a small context bonus to break
    ties when nearby lines collectively cover the event."""
    target_words = set(_normalize_words(target_text))
    if len(target_words) < 3:
        return None

    half = window_size // 2
    best_score = 0.0
    best_idx = -1
    for i in range(len(transcript_lines)):
        own_words = _normalize_words(transcript_lines[i]["text"])
        own_score = _score_overlap(target_words, own_words)
        lo = max(0, i - half)
        hi = min(len(transcript_lines), i + half + 1)
        ctx_words: List[str] = []
        for line in transcript_lines[lo:hi]:
            ctx_words.extend(_normalize_words(line["text"]))
        ctx_score = _score_overlap(target_words, ctx_words)
        # Centroid dominates; context is a tiebreaker
        combined = own_score * 2.0 + ctx_score * 0.5
        if combined > best_score:
            best_score = combined
            best_idx = i

    if best_idx < 0 or best_score < min_score:
        return None

    line = transcript_lines[best_idx]
    return {
        "seconds": line["seconds"],
        "file": line["file"],
        "score": best_score,
        "timestamp": _format_timestamp(line["seconds"]),
    }


def _extract_section(notes_md: str, section_name: str) -> Tuple[int, int, List[str]]:
    """Return (start_line, end_line, lines) of a ## section in markdown notes.
    end_line is exclusive."""
    lines = notes_md.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(f"## {section_name.lower()}"):
            start = i
            break
    if start < 0:
        return -1, -1, []
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return start, end, lines[start + 1 : end]


def anchor_notes(
    notes_md: str,
    raw_transcript_text: str,
    min_score: float = 1.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Annotate ## Key Events bullets with [HH:MM:SS] anchors.

    Returns (annotated_markdown, anchors_list). Each anchor is:
    {"text": ..., "timestamp": "HH:MM:SS", "seconds": int, "file": str, "score": float}
    """
    transcript_lines = parse_timestamped_transcript(raw_transcript_text)
    if not transcript_lines:
        return notes_md, []

    start, end, section_lines = _extract_section(notes_md, "Key Events")
    if start < 0:
        return notes_md, []

    anchors: List[Dict[str, Any]] = []
    new_section_lines: List[str] = []
    for raw_line in section_lines:
        line = raw_line.rstrip()
        # Recognize bullet lines
        bullet_match = re.match(r"^(\s*[-*+]\s+|\s*\d+\.\s+)(.*)$", line)
        if not bullet_match or not bullet_match.group(2).strip():
            new_section_lines.append(line)
            continue
        prefix, content = bullet_match.groups()
        # Skip if already anchored
        if re.match(r"^\[\d{2}:\d{2}:\d{2}\]", content):
            new_section_lines.append(line)
            continue
        anchor = find_anchor_for_text(content, transcript_lines, min_score=min_score)
        if anchor:
            new_section_lines.append(f"{prefix}[{anchor['timestamp']}] {content}")
            anchors.append({
                "text": content,
                "timestamp": anchor["timestamp"],
                "seconds": anchor["seconds"],
                "file": anchor["file"],
                "score": anchor["score"],
            })
        else:
            new_section_lines.append(line)

    all_lines = notes_md.splitlines()
    annotated = (
        all_lines[: start + 1] + new_section_lines + all_lines[end:]
    )
    return "\n".join(annotated), anchors
