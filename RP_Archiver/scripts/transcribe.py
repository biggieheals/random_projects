#!/usr/bin/env python
import json
from pathlib import Path
import typer
import whisperx

app = typer.Typer()

@app.command()
def run(audio: Path,
        out_json: Path = Path("transcripts/session.json"),
        model_name: str = "large-v3",
        device: str = "cuda",
        hf_token: str = typer.Option(None, envvar="HF_TOKEN"),
        batch_size: int = 16):
    """Transcribe + diarize one audio file using WhisperX."""
    audio = Path(audio)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    model = whisperx.load_model(model_name, device=device, compute_type="auto")

    print("Loading audio…")
    audio_data = whisperx.load_audio(str(audio))

    print("Transcribing…")
    result = model.transcribe(audio_data, batch_size=batch_size)

    print("Aligning timestamps…")
    result = whisperx.align(result["segments"], model, audio_data, device=device)

    print("Diarizing…")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    result = diarize_model(audio_data, result)

    json.dump(result, open(out_json, "w"), indent=2)
    print(f"Wrote {out_json}")

if __name__ == "__main__":