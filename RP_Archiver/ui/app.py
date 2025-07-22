import streamlit as st, json
from chromadb import Client
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="D&D Session Browser", layout="wide")
st.title("D&D Session Browser")

client = Client(path="../index_db")
col = client.get_collection("transcripts")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = st.text_input("Ask your notes:")
if query:
    q_emb = model.encode([query])[0]
    res = col.query(query_embeddings=[q_emb], n_results=15)
    st.subheader("Relevant lines")
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        st.markdown(f"**[{meta['start']:.1f}s] {meta['speaker']}**: {doc}")

st.subheader("Minutes")
try:
    data = json.load(open("../summaries/session_minutes.json"))
    for e in data:
        st.write(f"- **{e.get('timestamp','?')} – {e.get('EventType','?')}**: {e.get('short_summary','')}")
except FileNotFoundError:
    st.info("No minutes file yet. Run summarize_events.py.")