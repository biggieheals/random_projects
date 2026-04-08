# D&D Session Processor

Turn recorded D&D sessions into clean structured notes and a persistent campaign memory — local-first, with optional cross-session speaker recognition.

## What it does

1. **Transcribes** session audio with Whisper (faster-whisper preferred).
2. **Optional speaker diarization** via `pyannote.audio` — labels who said what, and remembers voices across future sessions.
3. **Cleans** the transcript (fillers, false starts) without losing game content.
4. **Generates structured session notes**: scenes, key events, NPCs, locations, loot, quests, consequences.
5. **Extracts structured entities** and merges them into a persistent campaign memory.
6. **Optional narrative campaign journal** for immersive reading.
7. **Optional Obsidian-friendly wiki** export.
8. **Campaign Q&A**: ask "What quests are still open?" or "When did we first meet Lord Brannis?"

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You also need **ffmpeg** on PATH:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: https://ffmpeg.org/download.html

### LLM backends (pick any, configure in Settings tab)

| Backend | Install | Notes |
|---|---|---|
| `fallback` | nothing | Always works. Cleanup + raw structure only. |
| `openai` | `pip install openai` | Default `gpt-4o-mini`. Also works for **OpenRouter, LM Studio, vLLM, Groq, Together** — just change the Base URL in Settings. |
| `anthropic` | `pip install anthropic` | Default `claude-sonnet-4-5`. |
| `ollama` | install [Ollama](https://ollama.com), `ollama pull llama3.1:8b` | Fully local. |

You can mix Whisper (local) with any LLM backend for the text part.

### Speaker diarization (optional)

```bash
pip install pyannote.audio torch
```

Then:
1. Get a free token at https://hf.co/settings/tokens
2. Visit and accept the model terms at:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/embedding
3. Paste the token into Settings → HF Token
4. Tick "Speaker diarization (pyannote)" in the Process tab

**How cross-session recognition works:** After diarization, the app extracts a voice embedding for each detected speaker. The first time a voice appears it asks you to name it ("Jess", "DM", "Ethan"). On the next session, it compares new voices to known ones via cosine similarity — if there's a match above the threshold (default 0.65, tune in Settings), it's auto-labelled. Voice prints are stored per-campaign in `speaker_registry.json` and refined over time as more samples accumulate. You can rename or remove voices anytime in the Speakers tab.

## Run

```bash
python main.py        # GUI
python main.py --cli --campaign mycampaign session1.mp3
python main.py --ask "What quests are still unresolved?" --campaign mycampaign
```

## Folder layout

```
campaigns/<campaign_name>/
  campaign_memory.json
  speaker_registry.json
  session_index.json
  wiki/
    npc_index.md  location_index.md  quest_log.md  item_log.md
    faction_index.md  session_index.md
  sessions/session_001_20260408_the_goblin_cave/
    raw_transcript.txt        ← with [HH:MM:SS] timestamps + speakers
    cleaned_transcript.txt
    session_notes.md / .txt
    campaign_journal.md
    extracted_entities.json
    pending_speakers.json     ← present until you name new voices
    metadata.json
```

## Troubleshooting

**"No Whisper backend could be loaded" but `python -c "import faster_whisper"` works?**
The new error message now shows the *actual* exception. The most common causes on Windows:
- **cuDNN DLL not found** → Set Device to `cpu` in the GUI, or install CUDA + cuDNN.
- **ctranslate2 DLL load failed** → `pip install --force-reinstall ctranslate2`
- **ffmpeg missing** → install ffmpeg and restart your terminal so PATH updates.

**Pyannote complains about model access?** You forgot to accept the terms on both `pyannote/speaker-diarization-3.1` *and* `pyannote/embedding`.

**Speaker recognition is mixing up voices?** Raise the match threshold in Settings (try 0.72). It's intentionally a bit loose by default so the same person isn't asked to be named twice.

## Future improvements

- Per-event timestamp anchors that link notes back to a second in the audio
- Vector search over session text for semantic queries
- Character sheets (HP/inventory) tracked across sessions
- Relationship graph (NPC ↔ faction ↔ location)
- Timeline view (Mermaid or HTML)
- Direct Obsidian vault sync
- Bulk reprocess past sessions when prompts improve
