#!/usr/bin/env python
import json, yaml
from pathlib import Path
import typer
from llama_cpp import Llama
from rich.progress import track

app = typer.Typer()

SYSTEM_PROMPT = (
    "You are a meticulous D&D session note-taker. "
    "Extract structured events from the transcript according to the given JSON schema. "
    "Return ONLY valid JSON (a list of event objects). Include timestamps."
)

@app.command()
def run(transcript: Path,
        schema_yaml: Path,
        out_json: Path = Path("summaries/session_minutes.json"),
        out_md: Path = Path("summaries/session_minutes.md"),
        model_path: Path = Path("models/llama-3-8b-instruct.Q4_K_M.gguf"),
        ctx_size: int = 4096,
        chunk_size: int = 60):
    trans = json.load(open(transcript))
    schema = yaml.safe_load(open(schema_yaml))

    llm = Llama(model_path=str(model_path), n_ctx=ctx_size, n_gpu_layers=-1, verbose=False)

    segs = trans["segments"]
    chunks = [segs[i:i+chunk_size] for i in range(0, len(segs), chunk_size)]
    events_all = []

    for chunk in track(chunks, description="Summarizing chunks"):
        text = "
".join(f"[{s['start']:.1f}] {s.get('speaker_name', s['speaker'])}: {s['text']}" for s in chunk)
        user_prompt = (
            f"Schema (JSON):
{json.dumps(schema, indent=2)}

"
            f"Transcript chunk:
{text}

"
            f"Return JSON list of events:"
        )
        resp = llm(prompt=f"<s>[INST]<<SYS>>{SYSTEM_PROMPT}<<SYS>>
{user_prompt}[/INST]", max_tokens=2048)
        raw = resp["choices"][0]["text"].strip()
        # simple guard: find first '[' and last ']'
        start_i, end_i = raw.find('['), raw.rfind(']')
        events_chunk = json.loads(raw[start_i:end_i+1])
        events_all.extend(events_chunk)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(events_all, open(out_json, "w"), indent=2)

    # Markdown minutes
    with open(out_md, "w") as f:
        f.write("# Session Minutes

")
        for e in events_all:
            f.write(f"- **{e.get('timestamp','?')} – {e.get('EventType','?')}**: {e.get('short_summary','')}
")
    print(f"Wrote {out_json} and {out_md}")

if __name__ == "__main__":
    app()