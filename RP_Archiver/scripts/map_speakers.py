#!/usr/bin/env python
import json
from pathlib import Path
import typer
import numpy as np
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Model
from pydub import AudioSegment

app = typer.Typer()

@app.command()
def run(transcript: Path,
        full_audio: Path,
        refs_dir: Path = Path("refs"),
        out_json: Path = Path("transcripts/session.mapped.json"),
        model_name: str = "pyannote/embedding",
        hf_token: str = typer.Option(None, envvar="HF_TOKEN")):
    """Assign real speaker names via embedding similarity.

    refs_dir should contain WAV files: Alice.wav, Bob.wav, GM.wav, etc.
    """
    transcript = json.load(open(transcript))

    # Load embedding model
    emb_model = Model.from_pretrained(model_name, use_auth_token=hf_token)

    def embed_audio_segment(audio_array, sr):
        # model expects float32 mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        return emb_model(audio_array, sample_rate=sr).detach().cpu().numpy()

    # Reference embeddings
    refs = {}
    for wav in refs_dir.glob("*.wav"):
        audio, sr = sf.read(wav)
        refs[wav.stem] = embed_audio_segment(audio, sr)

    # Helper to slice audio by seconds
    full = AudioSegment.from_file(full_audio)

    def slice_and_embed(start, end):
        seg = full[start*1000:end*1000]  # pydub works in ms
        samples = np.array(seg.get_array_of_samples()).astype(np.float32) / (2**15)
        # handle stereo
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels)).mean(axis=1)
        return embed_audio_segment(samples, seg.frame_rate)

    # Compute seg embeddings (optionally cache)
    for seg in transcript["segments"]:
        st, en = seg["start"], seg["end"]
        seg_emb = slice_and_embed(st, en)
        # cosine vs each ref
        sims = {name: float(cosine_similarity(seg_emb, ref_emb)[0][0]) for name, ref_emb in refs.items()}
        best = max(sims, key=sims.get)
        seg["speaker_name"] = best
        seg["speaker_scores"] = sims

    json.dump(transcript, open(out_json, "w"), indent=2)
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    app()