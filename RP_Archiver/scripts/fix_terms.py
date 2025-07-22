#!/usr/bin/env python
import re, json, yaml
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def run(inp: Path, gloss: Path, outp: Path = Path("transcripts/session.clean.json")):
    """Apply regex replacements from glossary.yaml to transcript JSON."""
    tran = json.load(open(inp))
    rep = yaml.safe_load(open(gloss))["replacements"]

    for seg in tran["segments"]:
        for bad, good in rep.items():
            seg["text"] = re.sub(bad, good, seg["text"], flags=re.IGNORECASE)

    json.dump(tran, open(outp, "w"), indent=2)
    print(f"Wrote {outp}")

if __name__ == "__main__":
    app()