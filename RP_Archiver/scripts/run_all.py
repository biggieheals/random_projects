#!/usr/bin/env python
import json, typer
from pathlib import Path

app = typer.Typer()

@app.command()
def to_markdown(minutes_json: Path, transcript_json: Path, out_md: Path):
    mins = json.load(open(minutes_json))
    trans = json.load(open(transcript_json))

    with open(out_md, "w") as f:
        f.write("---
")
        f.write(f"session: {out_md.stem}
")
        f.write("---

")
        f.write("# Minutes

")
        for e in mins:
            f.write(f"- **{e.get('timestamp','?')} – {e.get('EventType','?')}**: {e.get('short_summary','')}
")
        f.write("
# Transcript (excerpt)

")
        for s in trans["segments"][:200]:  # limit length or split files
            f.write(f"> **[{s['start']:.1f}] {s.get('speaker_name', s['speaker'])}**: {s['text']}

")
    print(f"Wrote {out_md}")

if __name__ == "__main__":
    app()