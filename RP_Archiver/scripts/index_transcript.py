#!/usr/bin/env python
import json
from pathlib import Path
import typer
from chromadb import Client
from sentence_transformers import SentenceTransformer

app = typer.Typer()

@app.command()
def run(inp: Path,
        dbdir: Path = Path("index_db"),
        collection: str = "transcripts",
        model_name: str = "all-MiniLM-L6-v2"):
    trans = json.load(open(inp))
    model = SentenceTransformer(model_name)

    client = Client(path=str(dbdir))
    col = client.get_or_create_collection(collection)

    texts, ids, metas = [], [], []
    for i, s in enumerate(trans["segments"]):
        texts.append(s["text"])
        ids.append(f"seg_{i}")
        metas.append({"start": s["start"], "end": s["end"], "speaker": s.get("speaker_name", s["speaker"])})

    embs = model.encode(texts, batch_size=64, convert_to_numpy=True)
    col.add(ids=ids, embeddings=embs, metadatas=metas, documents=texts)
    print(f"Indexed {len(texts)} segments into {dbdir}")

if __name__ == "__main__":
    app()