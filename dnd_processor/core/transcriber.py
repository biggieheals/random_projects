"""Whisper-based transcription with faster-whisper preferred, openai-whisper fallback.

Key fix vs earlier version: import-time and load-time errors are reported separately,
with the FULL exception text. No more silent "not installed" when the real cause is a
missing CUDA/cuDNN DLL or a ctranslate2 compatibility issue.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
import platform
import traceback


class TranscriptionError(Exception):
    pass


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Transcriber:
    """Wraps Whisper. Prefers faster-whisper for speed.

    Returns a list of segments: {"start", "end", "text", "file"}
    """

    def __init__(self, config, logger: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log = logger or (lambda m: None)
        self.backend = None
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        want = self.config.whisper_backend
        errors: List[str] = []

        # ---- Try faster-whisper ----
        if want in ("auto", "faster-whisper"):
            mod_ok = False
            try:
                from faster_whisper import WhisperModel  # noqa: F401
                mod_ok = True
                self.log("[transcribe] faster-whisper module imported OK.")
            except Exception as e:
                msg = f"faster-whisper import failed: {type(e).__name__}: {e}"
                self.log(f"[transcribe] {msg}")
                errors.append(msg)

            if mod_ok:
                try:
                    from faster_whisper import WhisperModel
                    device = self.config.whisper_device
                    if device == "auto":
                        device = "cpu"  # safe default; users can flip to cuda
                    compute = self.config.whisper_compute_type
                    if compute == "auto":
                        compute = "int8" if device == "cpu" else "float16"
                    self.log(
                        f"[transcribe] Loading faster-whisper '{self.config.whisper_model}' "
                        f"on {device} ({compute})..."
                    )
                    self._model = WhisperModel(
                        self.config.whisper_model,
                        device=device,
                        compute_type=compute,
                    )
                    self.backend = "faster-whisper"
                    self.log("[transcribe] faster-whisper model loaded.")
                    return
                except Exception as e:
                    full = traceback.format_exc(limit=3)
                    msg = f"faster-whisper model load failed: {type(e).__name__}: {e}"
                    self.log(f"[transcribe] {msg}")
                    self.log(full)
                    errors.append(msg)
                    if platform.system() == "Windows" and "cudnn" in str(e).lower():
                        errors.append(
                            "  HINT: cuDNN DLL not found. Set Device=cpu in the GUI, "
                            "or install CUDA+cuDNN."
                        )
                    elif "ctranslate2" in str(e).lower() or "DLL" in str(e):
                        errors.append(
                            "  HINT: ctranslate2 native library failed. Try: "
                            "pip install --force-reinstall ctranslate2"
                        )

        # ---- Try openai-whisper ----
        if want in ("auto", "openai-whisper"):
            mod_ok = False
            try:
                import whisper  # noqa: F401
                mod_ok = True
                self.log("[transcribe] openai-whisper module imported OK.")
            except Exception as e:
                msg = f"openai-whisper import failed: {type(e).__name__}: {e}"
                self.log(f"[transcribe] {msg}")
                errors.append(msg)

            if mod_ok:
                try:
                    import whisper
                    self.log(f"[transcribe] Loading openai-whisper '{self.config.whisper_model}'...")
                    self._model = whisper.load_model(self.config.whisper_model)
                    self.backend = "openai-whisper"
                    self.log("[transcribe] openai-whisper model loaded.")
                    return
                except Exception as e:
                    msg = f"openai-whisper model load failed: {type(e).__name__}: {e}"
                    self.log(f"[transcribe] {msg}")
                    errors.append(msg)

        raise TranscriptionError(
            "No Whisper backend could be loaded.\n  - " + "\n  - ".join(errors)
            + "\n\nFix options:\n"
            "  1. pip install faster-whisper  (recommended)\n"
            "  2. pip install openai-whisper\n"
            "  3. Set Device=cpu in the GUI if you have no CUDA/cuDNN\n"
            "  4. Make sure ffmpeg is installed and on PATH"
        )

    def transcribe_files(
        self,
        files: List[Path],
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Transcribe one or more files as a single chronological session."""
        self._load_model()
        all_segments: List[Dict[str, Any]] = []
        running_offset = 0.0
        per_file_offsets: List[Dict[str, Any]] = []

        for idx, f in enumerate(files):
            self.log(f"[transcribe] ({idx+1}/{len(files)}) {f.name}")
            if progress:
                progress(idx / max(len(files), 1), f"Transcribing {f.name}")

            segs, duration = self._transcribe_one(f)
            file_start = running_offset
            for s in segs:
                s["file"] = f.name
                s["start"] += running_offset
                s["end"] += running_offset
                all_segments.append(s)
            per_file_offsets.append({
                "file": f.name,
                "path": str(f),
                "start_offset": file_start,
                "duration": duration or 0.0,
            })
            running_offset += duration or 0.0

        if progress:
            progress(1.0, "Transcription complete")

        return {
            "segments": all_segments,
            "total_duration": running_offset,
            "backend": self.backend,
            "model": self.config.whisper_model,
            "files": [f.name for f in files],
            "file_offsets": per_file_offsets,
        }

    def _transcribe_one(self, path: Path):
        if self.backend == "faster-whisper":
            segments_gen, info = self._model.transcribe(
                str(path),
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            segs = []
            for s in segments_gen:
                segs.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text.strip(),
                })
            return segs, float(info.duration)
        else:
            result = self._model.transcribe(str(path), verbose=False)
            segs = []
            dur = 0.0
            for s in result.get("segments", []):
                segs.append({
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": s["text"].strip(),
                })
                dur = max(dur, float(s["end"]))
            return segs, dur


def segments_to_text(segments: List[Dict[str, Any]], include_timestamps: bool = True,
                     include_speakers: bool = True) -> str:
    lines = []
    current_file = None
    for s in segments:
        if s.get("file") and s["file"] != current_file:
            current_file = s["file"]
            lines.append(f"\n--- {current_file} ---\n")
        prefix_parts = []
        if include_timestamps:
            prefix_parts.append(f"[{_format_timestamp(s['start'])}]")
        if include_speakers and s.get("speaker"):
            prefix_parts.append(f"{s['speaker']}:")
        prefix = " ".join(prefix_parts)
        if prefix:
            lines.append(f"{prefix} {s['text']}")
        else:
            lines.append(s["text"])
    return "\n".join(lines)
