"""Speaker diarization via pyannote.audio.

Optional dependency: if pyannote isn't installed (or there's no HF token), the
Diarizer raises DiarizationUnavailable and the pipeline gracefully skips.

What this does per audio file:
1. Run pyannote/speaker-diarization-3.1 -> Annotation with SPEAKER_00, SPEAKER_01, ...
2. For each speaker, take their longest contiguous turn and extract a voice embedding
   via pyannote/embedding. One vector per speaker per file.
3. Return per-speaker turn lists + per-speaker embeddings.

The pipeline then:
- merges per-file results (a speaker can recur across files)
- matches each speaker against the campaign's SpeakerRegistry
- assigns speaker labels to whisper segments via maximum-overlap
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any


class DiarizationUnavailable(Exception):
    pass


class Diarizer:
    def __init__(self, hf_token: str, logger: Optional[Callable[[str], None]] = None,
                 device: str = "cpu"):
        self.hf_token = hf_token
        self.log = logger or (lambda m: None)
        self.device = device
        self._pipeline = None
        self._embedder = None
        self._Segment = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline, Inference, Model
            from pyannote.core import Segment
        except ImportError as e:
            raise DiarizationUnavailable(
                f"pyannote.audio is not installed: {e}\n"
                f"Install with: pip install pyannote.audio"
            )
        if not self.hf_token:
            raise DiarizationUnavailable(
                "HuggingFace token required. Get one at https://hf.co/settings/tokens "
                "and accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1"
            )
        try:
            self.log("[diarize] Loading pyannote/speaker-diarization-3.1...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            try:
                import torch
                if self.device == "cuda" and torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))
                    self.log("[diarize] Using CUDA.")
            except Exception:
                pass

            self.log("[diarize] Loading pyannote/embedding...")
            emb_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.hf_token,
            )
            self._embedder = Inference(emb_model, window="whole")
            self._Segment = Segment
            self.log("[diarize] Pyannote ready.")
        except Exception as e:
            raise DiarizationUnavailable(
                f"Failed to load pyannote models: {e}\n"
                f"Make sure you have accepted the terms at:\n"
                f"  https://hf.co/pyannote/speaker-diarization-3.1\n"
                f"  https://hf.co/pyannote/embedding"
            )

    def diarize_file(self, audio_path: Path) -> Dict[str, Any]:
        """Run diarization on one audio file.

        Returns:
        {
          "turns": [{"speaker": "SPEAKER_00", "start": 1.2, "end": 4.5}, ...],
          "embeddings": {"SPEAKER_00": [...], "SPEAKER_01": [...]}
        }
        """
        self._load()
        self.log(f"[diarize] Running diarization on {audio_path.name}...")
        diarization = self._pipeline(str(audio_path))

        # Collect turns per speaker
        speaker_turns: Dict[str, List[Tuple[float, float]]] = {}
        all_turns: List[Dict[str, Any]] = []
        for turn, _, label in diarization.itertracks(yield_label=True):
            speaker_turns.setdefault(label, []).append((turn.start, turn.end))
            all_turns.append({
                "speaker": label,
                "start": float(turn.start),
                "end": float(turn.end),
            })

        # Extract one embedding per speaker (use their longest single turn)
        embeddings: Dict[str, List[float]] = {}
        for label, turns in speaker_turns.items():
            longest = max(turns, key=lambda t: t[1] - t[0])
            try:
                seg = self._Segment(longest[0], longest[1])
                emb = self._embedder.crop(str(audio_path), seg)
                # emb is a numpy array
                if hasattr(emb, "tolist"):
                    vec = emb.tolist()
                    # Some pyannote versions return shape (1, D); flatten
                    if isinstance(vec, list) and vec and isinstance(vec[0], list):
                        vec = vec[0]
                    embeddings[label] = vec
                else:
                    embeddings[label] = list(emb)
            except Exception as e:
                self.log(f"[diarize] embedding extraction failed for {label}: {e}")

        self.log(f"[diarize] Found {len(speaker_turns)} speakers in {audio_path.name}.")
        return {"turns": all_turns, "embeddings": embeddings}


def assign_speakers_to_segments(
    whisper_segments: List[Dict[str, Any]],
    diarization_turns: List[Dict[str, Any]],
    label_map: Dict[str, str],
) -> None:
    """For each whisper segment, find the speaker with the most temporal overlap and
    write `segment["speaker"]` in place. label_map maps raw labels (SPEAKER_00) to
    display names (e.g. "Jess" or just "SPEAKER_00")."""
    if not diarization_turns:
        return
    # Bucket turns by raw label for speed
    by_label: Dict[str, List[Tuple[float, float]]] = {}
    for t in diarization_turns:
        by_label.setdefault(t["speaker"], []).append((t["start"], t["end"]))

    for seg in whisper_segments:
        s, e = seg["start"], seg["end"]
        best_label = None
        best_overlap = 0.0
        for label, turns in by_label.items():
            overlap = sum(max(0.0, min(e, te) - max(s, ts)) for ts, te in turns)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label:
            seg["speaker"] = label_map.get(best_label, best_label)
