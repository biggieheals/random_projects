"""All prompt templates for LLM stages. Edit here to tune behaviour."""

CLEANUP_PROMPT = """You are cleaning up a raw transcript of a live Dungeons & Dragons session.

GOAL: Produce a cleaner, more readable transcript that preserves the ACTUAL CONTENT of the game.

REMOVE OR REDUCE:
- Filler words (um, uh, like, you know, I mean) when they add no meaning
- Obvious false starts and repeated words ("I—I went—I went to the tavern" -> "I went to the tavern")
- Clearly off-topic table chatter (snack runs, real-world tangents, bathroom breaks) WHEN clearly identifiable
- Duplicate lines from transcription errors

DO NOT REMOVE OR ALTER:
- In-character dialogue and roleplay
- Dice rolls, mechanical decisions, combat actions and outcomes
- DM narration of the world, NPCs, and lore
- Player planning and discussion of game decisions
- Jokes or tangents that involve the game itself
- NPC names, place names, items, numbers

RULES:
- Keep the speaker structure and chronological order intact
- Do NOT summarise. This is cleanup, not summary.
- Do NOT invent content. If unsure, keep it.
- Preserve timestamps if present at the start of lines.

Here is the transcript chunk to clean:

---
{transcript}
---

Return ONLY the cleaned transcript text, no preamble or commentary."""


SESSION_NOTES_PROMPT = """You are an expert D&D campaign scribe. Generate structured session notes from this cleaned transcript.

CRITICAL RULES:
- Use ONLY information present in the transcript. Do NOT invent events, names, or outcomes.
- If a section has no content in the transcript, write "None noted" for that section.
- Prefer accuracy and traceability over dramatic prose.
- Keep names exactly as they appear in the transcript.
- If you are uncertain about a name or fact, mark it with "(?)".

{prior_context}

TRANSCRIPT:
---
{transcript}
---

Produce the notes in this exact Markdown structure:

# Session Title
(A short, factual title based on what happened)

## Short Summary
(3-6 sentences summarising what happened this session.)

## Scene-by-Scene Breakdown
(List each distinct scene or encounter with a short factual description. Use bullet points or numbered scenes.)

## Key Events
(Chronological bullet list of the important events that occurred.)

## Character Highlights
(What each player character did that mattered. Use their names if known.)

## Important NPCs
(List NPCs encountered with a one-line description of who they are and what they did this session.)

## Important Locations
(List locations visited with brief descriptions.)

## Loot / Rewards / Items
(Items, gold, magic items, or rewards gained. Write "None noted" if nothing.)

## Quests / Hooks / Unresolved Threads
(Active quests, new hooks, unresolved mysteries.)

## Consequences for Future Sessions
(What changed in the world, who is now angry/allied, what promises were made, what looms.)

Return ONLY the Markdown notes."""


ENTITY_EXTRACTION_PROMPT = """Extract structured campaign entities from this D&D session write-up.

RULES:
- Only extract entities that are clearly present in the notes.
- Do NOT invent entities or details.
- Use exact names as they appear.
- Keep descriptions short and factual.
- If unsure, omit.

SESSION NOTES:
---
{notes}
---

Return ONLY a valid JSON object with this exact shape:

{{
  "player_characters": [
    {{"name": "", "description": "", "notes": ""}}
  ],
  "npcs": [
    {{"name": "", "description": "", "role": "", "status": "", "location": ""}}
  ],
  "locations": [
    {{"name": "", "description": "", "type": ""}}
  ],
  "factions": [
    {{"name": "", "description": "", "disposition": ""}}
  ],
  "quests": [
    {{"name": "", "description": "", "status": "active"}}
  ],
  "items": [
    {{"name": "", "description": "", "owner": "", "type": ""}}
  ],
  "events": [
    {{"description": "", "significance": ""}}
  ],
  "secrets": [
    {{"description": ""}}
  ]
}}

Quest status must be one of: "active", "completed", "failed", "unknown".
Return ONLY the JSON. No markdown fences, no commentary."""


JOURNAL_PROMPT = """Rewrite these D&D session notes as an immersive campaign journal entry.

STRICT RULES:
- Do NOT invent events, NPCs, places, or outcomes. Every fact must come from the notes.
- Write in flowing narrative prose, past tense, third person (or first-person-plural if the notes clearly use "the party").
- Keep a tone that feels like a campaign chronicle — evocative but grounded.
- Preserve all names, locations, and key events.
- Length: roughly 400-900 words depending on how much happened.
- Do NOT add dialogue that was not in the notes.

SESSION NOTES:
---
{notes}
---

Return ONLY the narrative journal entry, starting with a title line (# Title)."""


MEMORY_MERGE_PROMPT = """You are resolving whether two campaign entities refer to the same thing.

EXISTING ENTITY:
{existing}

NEW ENTITY:
{new}

Are these the same entity? Consider name variations, nicknames, titles, and obvious typos.
Return ONLY "YES" or "NO"."""


QA_PROMPT = """You are answering a question about a D&D campaign using ONLY the provided campaign memory and session notes.

RULES:
- Answer ONLY from the provided context.
- If the answer is not in the context, say "I don't have that information in the campaign notes."
- Cite session numbers when relevant (e.g., "First appeared in Session 3").
- Be concise and factual.

CAMPAIGN CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""
