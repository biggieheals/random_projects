"""Session processing pipeline.

The full pipeline (process) does: transcribe -> diarize -> clean -> notes ->
extract -> journal -> memory -> wiki.

The text-only pipeline (reprocess) skips transcribe + diarize and reuses an
existing session's raw_transcript.txt and (optionally) cleaned_transcript.txt.
This lets you swap LLM backends without paying the Whisper cost again.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Optional, Dict, Any

from config import Config
from core.llm import LLM
from core.transcriber import Transcriber, segments_to_text
from core.cleaner import Cleaner
from core.summarizer import Summarizer
from core.extractor import Extractor
from core.memory import CampaignMemory
from core.journal import Journalist
from core.wiki import export_wiki
from core.speaker_registry import SpeakerRegistry
from core.vector_store import VectorStore
from core.anchors import anchor_notes


def _slugify(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    return text[:max_len] or "session"


def _extract_title(notes_md: str) -> str:
    for line in notes_md.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled Session"


def _extract_summary(notes_md: str) -> str:
    lines = notes_md.splitlines()
    in_summary = False
    out = []
    for line in lines:
        if line.strip().lower().startswith("## short summary"):
            in_summary = True
            continue
        if in_summary:
            if line.strip().startswith("## "):
                break
            out.append(line)
    return "\n".join(out).strip()


class SessionPipeline:
    def __init__(
        self,
        config: Config,
        campaign_name: str,
        logger: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.config = config
        self.campaign_name = campaign_name
        self.log = logger or (lambda m: print(m))
        self.progress = progress or (lambda p, m: None)

        self.campaign_dir = Path(config.campaign_root) / campaign_name
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.campaign_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.wiki_dir = self.campaign_dir / "wiki"
        self.memory_path = self.campaign_dir / "campaign_memory.json"
        self.speaker_registry_path = self.campaign_dir / "speaker_registry.json"

        self.llm = LLM(config, logger=self.log)
        self.transcriber = Transcriber(config, logger=self.log)
        self.cleaner = Cleaner(self.llm, config, logger=self.log)
        self.summarizer = Summarizer(self.llm, config, logger=self.log)
        self.extractor = Extractor(self.llm, logger=self.log)
        self.journalist = Journalist(self.llm, logger=self.log)
        self.memory = CampaignMemory(self.memory_path)
        self.speaker_registry = SpeakerRegistry(
            self.speaker_registry_path,
            default_threshold=config.speaker_match_threshold,
        )
        self.vector_store = VectorStore(self.campaign_dir, logger=self.log)

    # ================= FULL PIPELINE =================
    def process(
        self,
        audio_files: List[Path],
        generate_journal: bool = True,
        update_memory: bool = True,
        export_wiki_flag: bool = True,
        enable_diarization: Optional[bool] = None,
        enable_vector: bool = True,
        enable_anchors: bool = True,
    ) -> Dict[str, Any]:
        if not audio_files:
            raise ValueError("No audio files provided.")

        if enable_diarization is None:
            enable_diarization = self.config.enable_diarization

        t0 = datetime.now()
        session_number = self.memory.next_session_number()
        date_str = t0.strftime("%Y%m%d")
        session_folder = self.sessions_dir / f"session_{session_number:03d}_{date_str}"
        session_folder.mkdir(parents=True, exist_ok=True)

        self.log(f"[pipeline] Campaign: {self.campaign_name}")
        self.log(f"[pipeline] Session #{session_number} -> {session_folder}")
        self.log(f"[pipeline] Files: {[f.name for f in audio_files]}")

        # --- 1. Transcribe ---
        self.progress(0.0, "Starting transcription")
        trans_result = self.transcriber.transcribe_files(
            audio_files,
            progress=lambda p, m: self.progress(0.0 + p * 0.40, m),
        )
        segments = trans_result["segments"]

        # --- 1b. Diarization (optional) ---
        diarization_info, unknown_speakers = self._maybe_diarize(
            audio_files, segments, trans_result, enable_diarization
        )

        raw_text = segments_to_text(segments, include_timestamps=True, include_speakers=True)
        (session_folder / "raw_transcript.txt").write_text(raw_text, encoding="utf-8")
        self.log(f"[pipeline] Raw transcript saved ({len(raw_text)} chars).")

        # --- 1c. Save pending speakers JSON if any ---
        if unknown_speakers:
            pending = {
                "session_number": session_number,
                "speakers": [
                    {"label": u["label"], "embedding": u["embedding"], "name": ""}
                    for u in unknown_speakers
                ],
            }
            (session_folder / "pending_speakers.json").write_text(
                json.dumps(pending, indent=2), encoding="utf-8"
            )

        # --- Text stages ---
        result = self._run_text_stages(
            session_folder=session_folder,
            session_number=session_number,
            date_str=date_str,
            segments=segments,
            raw_text=raw_text,
            generate_journal=generate_journal,
            update_memory=update_memory,
            export_wiki_flag=export_wiki_flag,
            enable_vector=enable_vector,
            enable_anchors=enable_anchors,
            existing_cleaned=None,
            progress_start=0.55,
        )

        # --- Metadata ---
        metadata = {
            "campaign": self.campaign_name,
            "session_number": session_number,
            "title": result["title"],
            "processed_at": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.now() - t0).total_seconds(),
            "audio_files": [f.name for f in audio_files],
            "audio_paths": [str(f) for f in audio_files],
            "transcription": {
                "backend": trans_result["backend"],
                "model": trans_result["model"],
                "total_duration": trans_result["total_duration"],
            },
            "file_offsets": trans_result.get("file_offsets", []),
            "diarization": diarization_info,
            "llm_backend": self.llm.backend,
            "options": {
                "generate_journal": generate_journal,
                "update_memory": update_memory,
                "export_wiki": export_wiki_flag,
                "diarization": enable_diarization,
                "vector": enable_vector,
                "anchors": enable_anchors,
            },
            "entity_counts": result["entity_totals"],
            "anchor_count": len(result.get("anchors", [])),
        }
        (result["session_folder"] / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self.progress(1.0, "Done")
        self.log(f"[pipeline] Complete in {metadata['duration_seconds']:.1f}s.")

        return {
            "session_folder": str(result["session_folder"]),
            "campaign_dir": str(self.campaign_dir),
            "session_number": session_number,
            "title": result["title"],
            "metadata": metadata,
            "unknown_speakers": unknown_speakers,
        }

    # ================= REPROCESS =================
    def reprocess(
        self,
        session_folder: Path,
        generate_journal: bool = True,
        update_memory: bool = True,
        export_wiki_flag: bool = True,
        enable_vector: bool = True,
        enable_anchors: bool = True,
        reuse_cleaned: bool = False,
    ) -> Dict[str, Any]:
        """Re-run text stages on an existing session folder, skipping transcription.

        - Reads raw_transcript.txt (required) for anchoring.
        - If reuse_cleaned=True and cleaned_transcript.txt exists, skips the cleanup
          stage too (useful when you only changed prompts for notes/extraction).
        - Detects the existing session number from metadata.json or folder name so
          campaign memory updates merge correctly instead of duplicating.
        """
        session_folder = Path(session_folder)
        raw_path = session_folder / "raw_transcript.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw_transcript.txt not found in {session_folder}")

        t0 = datetime.now()
        raw_text = raw_path.read_text(encoding="utf-8")

        # Try to read existing metadata for session number / date / audio paths
        session_number = None
        date_str = t0.strftime("%Y%m%d")
        existing_meta: Dict[str, Any] = {}
        meta_path = session_folder / "metadata.json"
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                session_number = existing_meta.get("session_number")
                processed = existing_meta.get("processed_at", "")
                if processed:
                    date_str = processed[:10].replace("-", "")
            except Exception:
                pass

        # Fallback: parse folder name like session_002_20260408_xxx
        if session_number is None:
            m = re.match(r"session_(\d+)_(\d{8})", session_folder.name)
            if m:
                session_number = int(m.group(1))
                date_str = m.group(2)
            else:
                session_number = self.memory.next_session_number()

        self.log(f"[reprocess] Campaign: {self.campaign_name}")
        self.log(f"[reprocess] Session #{session_number} -> {session_folder}")
        self.log(f"[reprocess] Reusing raw transcript ({len(raw_text)} chars).")

        existing_cleaned = None
        if reuse_cleaned:
            cp = session_folder / "cleaned_transcript.txt"
            if cp.exists():
                existing_cleaned = cp.read_text(encoding="utf-8")
                self.log(f"[reprocess] Reusing cleaned transcript ({len(existing_cleaned)} chars).")

        # We don't have segments anymore, but the text stages don't need them — only
        # the raw text (for anchors) and the plain transcript (for cleaning).
        result = self._run_text_stages(
            session_folder=session_folder,
            session_number=session_number,
            date_str=date_str,
            segments=None,
            raw_text=raw_text,
            generate_journal=generate_journal,
            update_memory=update_memory,
            export_wiki_flag=export_wiki_flag,
            enable_vector=enable_vector,
            enable_anchors=enable_anchors,
            existing_cleaned=existing_cleaned,
            progress_start=0.05,
        )

        # Update metadata in place (preserve transcription/diarization info)
        new_meta = dict(existing_meta)
        new_meta.update({
            "campaign": self.campaign_name,
            "session_number": session_number,
            "title": result["title"],
            "reprocessed_at": datetime.utcnow().isoformat(),
            "reprocess_duration_seconds": (datetime.now() - t0).total_seconds(),
            "llm_backend": self.llm.backend,
            "entity_counts": result["entity_totals"],
            "anchor_count": len(result.get("anchors", [])),
        })
        opts = new_meta.setdefault("options", {})
        opts.update({
            "generate_journal": generate_journal,
            "update_memory": update_memory,
            "export_wiki": export_wiki_flag,
            "vector": enable_vector,
            "anchors": enable_anchors,
        })
        (result["session_folder"] / "metadata.json").write_text(
            json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self.progress(1.0, "Done")
        self.log(f"[reprocess] Complete in {(datetime.now() - t0).total_seconds():.1f}s.")
        return {
            "session_folder": str(result["session_folder"]),
            "session_number": session_number,
            "title": result["title"],
            "metadata": new_meta,
        }

    # ================= SHARED TEXT STAGES =================
    def _run_text_stages(
        self,
        session_folder: Path,
        session_number: int,
        date_str: str,
        segments: Optional[List[Dict[str, Any]]],
        raw_text: str,
        generate_journal: bool,
        update_memory: bool,
        export_wiki_flag: bool,
        enable_vector: bool,
        enable_anchors: bool,
        existing_cleaned: Optional[str],
        progress_start: float,
    ) -> Dict[str, Any]:
        # --- Cleanup ---
        if existing_cleaned is not None:
            cleaned = existing_cleaned
            self.progress(progress_start + 0.05, "Reused cleaned transcript")
        else:
            self.progress(progress_start, "Cleaning transcript")
            if segments is not None:
                plain_text = segments_to_text(segments, include_timestamps=False, include_speakers=True)
            else:
                # Strip [HH:MM:SS] timestamps from raw text for cleaning
                plain_text = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", raw_text, flags=re.MULTILINE)
            cleaned = self.cleaner.clean(
                plain_text,
                progress=lambda p, m: self.progress(progress_start + p * 0.10, m),
            )
            (session_folder / "cleaned_transcript.txt").write_text(cleaned, encoding="utf-8")
            self.log(f"[pipeline] Cleaned transcript saved ({len(cleaned)} chars).")

        # --- Notes ---
        self.progress(progress_start + 0.10, "Generating session notes")
        prior_context = self.memory.get_prior_context() if update_memory else ""
        notes = self.summarizer.generate_notes(
            cleaned,
            prior_context=prior_context,
            progress=lambda p, m: self.progress(progress_start + 0.10 + p * 0.18, m),
        )

        # --- Anchors ---
        anchors: List[Dict[str, Any]] = []
        if enable_anchors and raw_text:
            try:
                notes, anchors = anchor_notes(notes, raw_text)
                if anchors:
                    self.log(f"[anchors] Anchored {len(anchors)} key event(s) to timestamps.")
            except Exception as e:
                self.log(f"[anchors] Failed: {e}")

        (session_folder / "session_notes.md").write_text(notes, encoding="utf-8")
        (session_folder / "session_notes.txt").write_text(notes, encoding="utf-8")
        title = _extract_title(notes)
        summary = _extract_summary(notes)

        if anchors:
            (session_folder / "anchors.json").write_text(
                json.dumps(anchors, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        self.log(f"[pipeline] Notes saved. Title: {title}")

        # Rename folder once we have a title (only if not already named)
        if "_" in session_folder.name:
            parts = session_folder.name.split("_")
            # session_002_20260408 -> 3 parts; with title slug, more parts
            if len(parts) <= 3:
                target = self.sessions_dir / f"session_{session_number:03d}_{date_str}_{_slugify(title)}"
                if target != session_folder and not target.exists():
                    try:
                        session_folder.rename(target)
                        session_folder = target
                    except Exception:
                        pass

        # --- Entity extraction ---
        self.progress(progress_start + 0.30, "Extracting entities")
        entities = self.extractor.extract(notes)
        (session_folder / "extracted_entities.json").write_text(
            json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        entity_totals = {k: len(v) for k, v in entities.items()}
        self.log(f"[pipeline] Extracted entities: {entity_totals}")

        # --- Journal ---
        if generate_journal:
            self.progress(progress_start + 0.34, "Writing campaign journal")
            journal = self.journalist.rewrite(notes)
            (session_folder / "campaign_journal.md").write_text(journal, encoding="utf-8")
            self.log("[pipeline] Campaign journal saved.")

        # --- Memory update ---
        if update_memory:
            self.progress(progress_start + 0.38, "Updating campaign memory")
            merge_stats = self.memory.merge_entities(entities, session_number)
            self.memory.record_session(
                session_number=session_number,
                title=title,
                folder=str(session_folder.relative_to(self.campaign_dir)),
                summary=summary,
            )
            self.memory.save()
            self.log(f"[pipeline] Memory updated. Merge stats: {merge_stats}")

            session_index = {"sessions": self.memory.data.get("sessions", [])}
            (self.campaign_dir / "session_index.json").write_text(
                json.dumps(session_index, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # --- Wiki ---
        if export_wiki_flag and update_memory:
            self.progress(progress_start + 0.41, "Exporting wiki")
            written = export_wiki(self.memory.data, self.wiki_dir)
            self.log(f"[pipeline] Wiki exported: {len(written)} files.")

        # --- Vector indexing ---
        if enable_vector:
            self.progress(progress_start + 0.43, "Indexing for semantic search")
            try:
                added = self.vector_store.index_session(session_number, title, cleaned)
                if added:
                    self.log(f"[vector] Indexed {added} chunks for session {session_number}.")
                elif not self.vector_store.available:
                    self.log("[vector] Skipped (chromadb not installed).")
            except Exception as e:
                self.log(f"[vector] Indexing error: {e}")

        return {
            "session_folder": session_folder,
            "title": title,
            "summary": summary,
            "entity_totals": entity_totals,
            "anchors": anchors,
        }

    # ================= DIARIZATION =================
    def _maybe_diarize(self, audio_files, segments, trans_result, enabled):
        if not enabled:
            return {"enabled": False}, []
        try:
            from core.diarizer import Diarizer, assign_speakers_to_segments
            from core.speaker_registry import cosine_similarity

            self.progress(0.40, "Running speaker diarization")
            diarizer = Diarizer(
                hf_token=self.config.hf_token,
                logger=self.log,
                device=self.config.diarization_device,
            )
            merged_turns: List[Dict[str, Any]] = []
            candidates: List[Dict[str, Any]] = []
            file_offsets = {fo["file"]: fo for fo in trans_result["file_offsets"]}

            for audio_file in audio_files:
                fo = file_offsets.get(audio_file.name, {})
                offset = fo.get("start_offset", 0.0)
                file_diar = diarizer.diarize_file(audio_file)
                for turn in file_diar["turns"]:
                    merged_turns.append({
                        "speaker": f"{audio_file.name}::{turn['speaker']}",
                        "start": turn["start"] + offset,
                        "end": turn["end"] + offset,
                    })
                for raw_label, emb in file_diar["embeddings"].items():
                    candidates.append({
                        "key": f"{audio_file.name}::{raw_label}",
                        "embedding": emb,
                        "file": audio_file.name,
                        "raw_label": raw_label,
                    })

            label_map: Dict[str, str] = {}
            self_groups: List[Dict[str, Any]] = []
            unknown_speakers: List[Dict[str, Any]] = []

            for cand in candidates:
                name, sim = self.speaker_registry.match(cand["embedding"])
                if name is None:
                    best_group = None
                    best_sim = 0.0
                    for g in self_groups:
                        s = cosine_similarity(cand["embedding"], g["embedding"])
                        if s > best_sim:
                            best_sim = s
                            best_group = g
                    if best_group is not None and best_sim >= self.speaker_registry.threshold:
                        best_group["members"].append(cand["key"])
                        label_map[cand["key"]] = best_group["name"]
                    else:
                        new_label = f"SPEAKER_{len(self_groups):02d}"
                        self_groups.append({
                            "name": new_label,
                            "embedding": cand["embedding"],
                            "members": [cand["key"]],
                        })
                        label_map[cand["key"]] = new_label
                        unknown_speakers.append({
                            "label": new_label,
                            "embedding": cand["embedding"],
                            "sample_file": cand["file"],
                        })
                else:
                    label_map[cand["key"]] = name
                    self.log(f"[diarize] Recognised {cand['raw_label']} in "
                             f"{cand['file']} as known '{name}' (sim={sim:.2f})")

            assign_speakers_to_segments(segments, merged_turns, label_map)
            info = {
                "enabled": True,
                "speakers_found": len(set(label_map.values())),
                "unknown_count": len(unknown_speakers),
            }
            self.log(f"[diarize] Done. {len(unknown_speakers)} new voice(s) need labelling.")
            return info, unknown_speakers
        except Exception as e:
            import traceback
            self.log(f"[diarize] Skipped: {e}")
            self.log(traceback.format_exc(limit=2))
            return {"enabled": False, "error": str(e)}, []

    # ================= SPEAKER COMMIT =================
    def commit_speaker_names(self, session_folder: Path,
                             assignments: Dict[str, str]) -> None:
        pending_path = session_folder / "pending_speakers.json"
        if not pending_path.exists():
            return
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        session_number = pending.get("session_number", 0)

        rename_map: Dict[str, str] = {}
        for sp in pending.get("speakers", []):
            label = sp["label"]
            new_name = assignments.get(label, "").strip()
            if not new_name:
                continue
            rename_map[label] = new_name
            self.speaker_registry.add_or_update(
                new_name, sp.get("embedding", []), session_number
            )
        self.speaker_registry.save()

        for fname in ("raw_transcript.txt", "cleaned_transcript.txt"):
            p = session_folder / fname
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            for old, new in rename_map.items():
                text = re.sub(rf"\b{re.escape(old)}\b", new, text)
            p.write_text(text, encoding="utf-8")

        pending_path.unlink(missing_ok=True)
        self.log(f"[speakers] Committed {len(rename_map)} speaker name(s).")

    # ================= SESSION DISCOVERY =================
    def list_sessions(self) -> List[Dict[str, Any]]:
        """Find all session folders in the campaign that have a raw_transcript.txt."""
        out = []
        if not self.sessions_dir.exists():
            return out
        for child in sorted(self.sessions_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "raw_transcript.txt").exists():
                continue
            info = {"folder": child, "name": child.name}
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    info["session_number"] = meta.get("session_number")
                    info["title"] = meta.get("title", "")
                    info["llm_backend"] = meta.get("llm_backend", "")
                except Exception:
                    pass
            out.append(info)
        return out
