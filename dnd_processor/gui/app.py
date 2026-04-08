"""Tkinter GUI for the D&D session processor."""
from __future__ import annotations
import os
import sys
import json
import queue
import threading
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import Config, CONFIG_PATH
from core.pipeline import SessionPipeline
from core.memory import CampaignMemory
from core.qa import CampaignQA
from core.llm import LLM
from core.speaker_registry import SpeakerRegistry
from core.vector_store import VectorStore


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".aac", ".wma", ".webm")
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
LLM_BACKENDS = ["fallback", "openai", "anthropic", "ollama"]


def _format_duration(path: Path) -> str:
    try:
        from mutagen import File as MFile
        f = MFile(str(path))
        if f is not None and f.info and getattr(f.info, "length", None):
            total = int(f.info.length)
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
    except Exception:
        pass
    try:
        mb = path.stat().st_size / (1024 * 1024)
        return f"{mb:.1f} MB"
    except Exception:
        return "?"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("D&D Session Processor")
        self.root.geometry("1040x820")
        self.root.minsize(880, 660)

        self.config = Config.load()
        self.selected_files: List[Path] = []
        self.log_queue: queue.Queue = queue.Queue()
        self.processing = False
        self.last_output_folder: Optional[Path] = None
        self.last_pipeline: Optional[SessionPipeline] = None
        self.last_unknown_speakers: List[Dict] = []

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.process_frame = ttk.Frame(nb)
        self.reprocess_frame = ttk.Frame(nb)
        self.query_frame = ttk.Frame(nb)
        self.anchors_frame = ttk.Frame(nb)
        self.speakers_frame = ttk.Frame(nb)
        self.settings_frame = ttk.Frame(nb)

        nb.add(self.process_frame, text="Process Session")
        nb.add(self.reprocess_frame, text="Reprocess")
        nb.add(self.query_frame, text="Campaign Q&A")
        nb.add(self.anchors_frame, text="Event Anchors")
        nb.add(self.speakers_frame, text="Speakers")
        nb.add(self.settings_frame, text="Settings")

        self._build_process_tab()
        self._build_reprocess_tab()
        self._build_query_tab()
        self._build_anchors_tab()
        self._build_speakers_tab()
        self._build_settings_tab()

    # ============= PROCESS TAB =============
    def _build_process_tab(self):
        f = self.process_frame

        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text="Campaign:").pack(side="left")
        self.campaign_var = tk.StringVar(value=self.config.default_campaign)
        ttk.Entry(top, textvariable=self.campaign_var, width=30).pack(side="left", padx=6)

        list_frame = ttk.LabelFrame(f, text="Audio Files")
        list_frame.pack(fill="both", expand=False, padx=10, pady=6)

        btn_row = ttk.Frame(list_frame)
        btn_row.pack(fill="x", padx=6, pady=6)
        ttk.Button(btn_row, text="Add Files...", command=self.add_files).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self.clear_files).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Move Up", command=lambda: self._move_file(-1)).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Move Down", command=lambda: self._move_file(1)).pack(side="left", padx=4)
        ttk.Label(btn_row, text="  (multiple files = one chronological session)").pack(side="left")

        cols = ("file", "size_duration")
        self.file_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=6)
        self.file_tree.heading("file", text="File")
        self.file_tree.heading("size_duration", text="Duration / Size")
        self.file_tree.column("file", width=620, anchor="w")
        self.file_tree.column("size_duration", width=140, anchor="w")
        self.file_tree.pack(fill="x", padx=6, pady=(0, 6))

        out_frame = ttk.Frame(f)
        out_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(out_frame, text="Campaigns root:").pack(side="left")
        self.output_var = tk.StringVar(value=self.config.campaign_root)
        ttk.Entry(out_frame, textvariable=self.output_var, width=60).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(out_frame, text="Browse...", command=self.choose_output).pack(side="left")

        opts = ttk.LabelFrame(f, text="Options")
        opts.pack(fill="x", padx=10, pady=6)

        row1 = ttk.Frame(opts)
        row1.pack(fill="x", padx=6, pady=4)
        ttk.Label(row1, text="Whisper:").pack(side="left")
        self.whisper_var = tk.StringVar(value=self.config.whisper_model)
        ttk.Combobox(row1, textvariable=self.whisper_var, values=WHISPER_MODELS,
                     state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(row1, text="Device:").pack(side="left", padx=(16, 0))
        self.device_var = tk.StringVar(value=self.config.whisper_device)
        ttk.Combobox(row1, textvariable=self.device_var, values=["cpu", "cuda", "auto"],
                     state="readonly", width=8).pack(side="left", padx=6)
        ttk.Label(row1, text="LLM:").pack(side="left", padx=(16, 0))
        self.llm_var = tk.StringVar(value=self.config.llm_backend)
        ttk.Combobox(row1, textvariable=self.llm_var, values=LLM_BACKENDS,
                     state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(row1, text="(configure in Settings)").pack(side="left")

        row2 = ttk.Frame(opts)
        row2.pack(fill="x", padx=6, pady=4)
        self.journal_var = tk.BooleanVar(value=True)
        self.memory_var = tk.BooleanVar(value=True)
        self.wiki_var = tk.BooleanVar(value=True)
        self.diarize_var = tk.BooleanVar(value=self.config.enable_diarization)
        self.vector_var = tk.BooleanVar(value=True)
        self.anchor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Journal", variable=self.journal_var).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Update memory", variable=self.memory_var).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Wiki", variable=self.wiki_var).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Diarization", variable=self.diarize_var).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Vector index", variable=self.vector_var).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Timestamp anchors", variable=self.anchor_var).pack(side="left", padx=4)

        actions = ttk.Frame(f)
        actions.pack(fill="x", padx=10, pady=6)
        self.start_btn = ttk.Button(actions, text="Start Processing", command=self.start_processing)
        self.start_btn.pack(side="left")
        ttk.Button(actions, text="Open Output Folder", command=self.open_output).pack(side="left", padx=6)

        prog_frame = ttk.Frame(f)
        prog_frame.pack(fill="x", padx=10, pady=(6, 2))
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(prog_frame, textvariable=self.status_var).pack(anchor="w", pady=(2, 0))

        log_frame = ttk.LabelFrame(f, text="Status Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled",
                                bg="#111418", fg="#d0d6de", insertbackground="#d0d6de")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ============= REPROCESS TAB =============
    def _build_reprocess_tab(self):
        f = self.reprocess_frame
        ttk.Label(
            f,
            text="Reprocess an existing session without re-transcribing. Use this when you "
                 "switch LLM backends, tweak prompts, or just want to re-extract entities and "
                 "regenerate notes from a transcript you already have.",
            wraplength=940, foreground="#666", justify="left",
        ).pack(padx=10, pady=(10, 6), anchor="w")

        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=4)
        ttk.Label(top, text="Campaign:").pack(side="left")
        self.rp_campaign_var = tk.StringVar(value=self.config.default_campaign)
        ttk.Entry(top, textvariable=self.rp_campaign_var, width=30).pack(side="left", padx=6)
        ttk.Button(top, text="Load Sessions", command=self._load_reprocess_sessions).pack(side="left")

        list_frame = ttk.LabelFrame(f, text="Existing sessions in this campaign")
        list_frame.pack(fill="both", expand=True, padx=10, pady=6)
        cols = ("num", "title", "folder", "llm")
        self.rp_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=10)
        for c, w in [("num", 60), ("title", 360), ("folder", 380), ("llm", 100)]:
            self.rp_tree.heading(c, text=c.title())
            self.rp_tree.column(c, width=w, anchor="w")
        self.rp_tree.pack(fill="both", expand=True, padx=4, pady=4)

        opts = ttk.LabelFrame(f, text="Reprocess options")
        opts.pack(fill="x", padx=10, pady=6)
        row = ttk.Frame(opts)
        row.pack(fill="x", padx=6, pady=4)

        ttk.Label(row, text="LLM:").pack(side="left")
        self.rp_llm_var = tk.StringVar(value=self.config.llm_backend)
        ttk.Combobox(row, textvariable=self.rp_llm_var, values=LLM_BACKENDS,
                     state="readonly", width=12).pack(side="left", padx=6)

        self.rp_journal_var = tk.BooleanVar(value=True)
        self.rp_memory_var = tk.BooleanVar(value=True)
        self.rp_wiki_var = tk.BooleanVar(value=True)
        self.rp_vector_var = tk.BooleanVar(value=True)
        self.rp_anchor_var = tk.BooleanVar(value=True)
        self.rp_reuse_clean_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Journal", variable=self.rp_journal_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Memory", variable=self.rp_memory_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Wiki", variable=self.rp_wiki_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Vector", variable=self.rp_vector_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Anchors", variable=self.rp_anchor_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Reuse cleaned (skip cleanup)",
                        variable=self.rp_reuse_clean_var).pack(side="left", padx=4)

        actions = ttk.Frame(f)
        actions.pack(fill="x", padx=10, pady=6)
        self.rp_btn = ttk.Button(actions, text="Reprocess Selected Session",
                                  command=self.start_reprocess)
        self.rp_btn.pack(side="left")

    def _load_reprocess_sessions(self):
        for i in self.rp_tree.get_children():
            self.rp_tree.delete(i)
        name = self.rp_campaign_var.get().strip() or "default"
        try:
            self.config.campaign_root = self.output_var.get()
            pipeline = SessionPipeline(self.config, campaign_name=name, logger=lambda m: None)
            sessions = pipeline.list_sessions()
            for s in sessions:
                self.rp_tree.insert("", "end", values=(
                    s.get("session_number", "?"),
                    s.get("title", ""),
                    s["name"],
                    s.get("llm_backend", ""),
                ), tags=(str(s["folder"]),))
            if not sessions:
                messagebox.showinfo("Reprocess", f"No sessions found in campaign '{name}'.")
        except Exception as e:
            messagebox.showerror("Reprocess", f"Failed to load sessions: {e}")

    def start_reprocess(self):
        if self.processing:
            return
        sel = self.rp_tree.selection()
        if not sel:
            messagebox.showwarning("Reprocess", "Select a session first.")
            return
        item = self.rp_tree.item(sel[0])
        folder_str = item["tags"][0] if item.get("tags") else None
        if not folder_str:
            return
        campaign_name = self.rp_campaign_var.get().strip() or "default"
        self.config.llm_backend = self.rp_llm_var.get()
        self.config.campaign_root = self.output_var.get()

        self.processing = True
        self.rp_btn.configure(state="disabled")
        self.progress["value"] = 0
        self._set_status("Reprocessing...")
        self._log(f"=== Reprocessing session in '{campaign_name}' ===")

        threading.Thread(
            target=self._run_reprocess,
            args=(campaign_name, Path(folder_str)),
            daemon=True,
        ).start()

    def _run_reprocess(self, campaign_name: str, folder: Path):
        try:
            pipeline = SessionPipeline(
                self.config,
                campaign_name=campaign_name,
                logger=self._log,
                progress=self._progress,
            )
            self.last_pipeline = pipeline
            result = pipeline.reprocess(
                folder,
                generate_journal=self.rp_journal_var.get(),
                update_memory=self.rp_memory_var.get(),
                export_wiki_flag=self.rp_wiki_var.get(),
                enable_vector=self.rp_vector_var.get(),
                enable_anchors=self.rp_anchor_var.get(),
                reuse_cleaned=self.rp_reuse_clean_var.get(),
            )
            self.last_output_folder = Path(result["session_folder"])
            self._log(f"✓ Reprocess done. Output: {result['session_folder']}")
            self._set_status(f"Reprocessed: {result['title']}")
        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}")
            self._log(traceback.format_exc())
            self._set_status(f"Error: {e}")
        finally:
            self.processing = False
            self.root.after(0, lambda: self.rp_btn.configure(state="normal"))

    # ============= QUERY TAB =============
    def _build_query_tab(self):
        f = self.query_frame
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Label(top, text="Campaign:").pack(side="left")
        self.qa_campaign_var = tk.StringVar(value=self.config.default_campaign)
        ttk.Entry(top, textvariable=self.qa_campaign_var, width=30).pack(side="left", padx=6)
        ttk.Button(top, text="Load", command=self._load_qa).pack(side="left")

        ttk.Label(top, text="Mode:").pack(side="left", padx=(16, 0))
        self.qa_mode_var = tk.StringVar(value="auto")
        ttk.Combobox(top, textvariable=self.qa_mode_var,
                     values=["auto", "structured", "semantic"],
                     state="readonly", width=12).pack(side="left", padx=4)
        self.qa_status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.qa_status_var, foreground="#888").pack(side="left", padx=10)

        q_frame = ttk.Frame(f)
        q_frame.pack(fill="x", padx=10)
        ttk.Label(q_frame, text="Ask a question about your campaign:").pack(anchor="w")
        self.qa_entry = ttk.Entry(q_frame)
        self.qa_entry.pack(fill="x", pady=4)
        self.qa_entry.bind("<Return>", lambda e: self._ask_qa())
        ttk.Button(q_frame, text="Ask", command=self._ask_qa).pack(anchor="e")

        out_frame = ttk.LabelFrame(f, text="Answer")
        out_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.qa_output = tk.Text(out_frame, wrap="word", state="disabled",
                                 bg="#111418", fg="#d0d6de")
        self.qa_output.pack(fill="both", expand=True, padx=4, pady=4)

        self._qa_memory: Optional[CampaignMemory] = None
        self._qa_vector: Optional[VectorStore] = None

    def _load_qa(self):
        name = self.qa_campaign_var.get().strip() or "default"
        campaign_dir = Path(self.config.campaign_root) / name
        mem_path = campaign_dir / "campaign_memory.json"
        if not mem_path.exists():
            messagebox.showinfo("Load", f"No memory file found at:\n{mem_path}")
            return
        self._qa_memory = CampaignMemory(mem_path)
        self._qa_vector = VectorStore(campaign_dir, logger=lambda m: None)
        vstats = self._qa_vector.stats()
        vec_msg = (f"vector: {vstats['count']} chunks indexed"
                   if vstats["available"] else "vector: not installed")
        self.qa_status_var.set(
            f"Loaded '{name}' — {self._qa_memory.session_count} session(s), {vec_msg}"
        )
        self._qa_write(f"Loaded campaign '{name}'.\nAsk anything about your campaign below.\n"
                       f"Tip: questions like 'what happened when...' use semantic search "
                       f"over transcript text; 'who is X' / 'when did we...' use structured memory.")

    def _ask_qa(self):
        if self._qa_memory is None:
            self._load_qa()
            if self._qa_memory is None:
                return
        q = self.qa_entry.get().strip()
        if not q:
            return
        llm = LLM(self.config, logger=lambda m: None)
        qa = CampaignQA(llm, self._qa_memory.data, vector_store=self._qa_vector)
        mode = self.qa_mode_var.get()
        self._qa_write(f"\nQ: {q}\n\n...thinking...")

        def worker():
            try:
                answer = qa.ask(q, mode=mode)
            except Exception as e:
                answer = f"Error: {e}"
            self.root.after(0, lambda: self._qa_write(f"\nQ: {q}\n\nA: {answer}\n", replace_last=True))
        threading.Thread(target=worker, daemon=True).start()
        self.qa_entry.delete(0, "end")

    def _qa_write(self, msg: str, replace_last: bool = False):
        self.qa_output.configure(state="normal")
        if replace_last:
            content = self.qa_output.get("1.0", "end")
            idx = content.rfind("...thinking...")
            if idx != -1:
                line = content[:idx].count("\n") + 1
                self.qa_output.delete(f"{line}.0", "end")
        self.qa_output.insert("end", msg + "\n")
        self.qa_output.see("end")
        self.qa_output.configure(state="disabled")

    # ============= ANCHORS TAB =============
    def _build_anchors_tab(self):
        f = self.anchors_frame
        ttk.Label(
            f,
            text="Click an event to jump to that timestamp in the source audio. "
                 "Anchors are auto-generated from the Key Events section by matching against "
                 "the timestamped raw transcript.",
            wraplength=940, foreground="#666",
        ).pack(padx=10, pady=(10, 6), anchor="w")

        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=4)
        ttk.Label(top, text="Campaign:").pack(side="left")
        self.an_campaign_var = tk.StringVar(value=self.config.default_campaign)
        ttk.Entry(top, textvariable=self.an_campaign_var, width=30).pack(side="left", padx=6)
        ttk.Button(top, text="Load Sessions", command=self._load_anchor_sessions).pack(side="left")

        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.LabelFrame(body, text="Sessions")
        body.add(left, weight=1)
        self.an_session_list = tk.Listbox(left)
        self.an_session_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.an_session_list.bind("<<ListboxSelect>>", lambda e: self._show_session_anchors())

        right = ttk.LabelFrame(body, text="Anchored events")
        body.add(right, weight=3)
        cols = ("ts", "event")
        self.an_tree = ttk.Treeview(right, columns=cols, show="headings")
        self.an_tree.heading("ts", text="Time")
        self.an_tree.heading("event", text="Event")
        self.an_tree.column("ts", width=80, anchor="w")
        self.an_tree.column("event", width=600, anchor="w")
        self.an_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.an_tree.bind("<Double-1>", lambda e: self._open_anchor())

        ttk.Button(f, text="Open Selected at Timestamp", command=self._open_anchor).pack(pady=(0, 8))

        self._anchor_sessions: List[Dict[str, Any]] = []
        self._current_anchors: List[Dict[str, Any]] = []
        self._current_session_meta: Optional[Dict[str, Any]] = None

    def _load_anchor_sessions(self):
        self.an_session_list.delete(0, "end")
        self.an_tree.delete(*self.an_tree.get_children())
        name = self.an_campaign_var.get().strip() or "default"
        try:
            pipeline = SessionPipeline(self.config, campaign_name=name, logger=lambda m: None)
            sessions = pipeline.list_sessions()
            self._anchor_sessions = sessions
            for s in sessions:
                self.an_session_list.insert(
                    "end",
                    f"S{s.get('session_number','?'):>3} — {s.get('title','(untitled)')}",
                )
        except Exception as e:
            messagebox.showerror("Anchors", f"Failed: {e}")

    def _show_session_anchors(self):
        sel = self.an_session_list.curselection()
        if not sel:
            return
        idx = sel[0]
        session = self._anchor_sessions[idx]
        folder = session["folder"]
        anchors_path = folder / "anchors.json"
        meta_path = folder / "metadata.json"
        self._current_anchors = []
        self._current_session_meta = None
        self.an_tree.delete(*self.an_tree.get_children())

        if meta_path.exists():
            try:
                self._current_session_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if not anchors_path.exists():
            self.an_tree.insert("", "end", values=("", "(no anchors for this session)"))
            return
        try:
            self._current_anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.an_tree.insert("", "end", values=("", f"(failed to load: {e})"))
            return

        for a in self._current_anchors:
            self.an_tree.insert("", "end", values=(a.get("timestamp", ""), a.get("text", "")))

    def _open_anchor(self):
        sel = self.an_tree.selection()
        if not sel or not self._current_anchors:
            return
        idx = self.an_tree.index(sel[0])
        if idx >= len(self._current_anchors):
            return
        anchor = self._current_anchors[idx]
        self._open_audio_at(anchor)

    def _open_audio_at(self, anchor: Dict[str, Any]):
        # Find the audio file path from session metadata
        audio_path = None
        if self._current_session_meta:
            paths = self._current_session_meta.get("audio_paths", [])
            file_offsets = self._current_session_meta.get("file_offsets", [])
            target_file = anchor.get("file")
            target_seconds = anchor.get("seconds", 0)
            local_seconds = target_seconds

            # If anchor file is known, use it
            if target_file and paths:
                for p in paths:
                    if Path(p).name == target_file:
                        audio_path = Path(p)
                        # Adjust to per-file local time
                        for fo in file_offsets:
                            if fo.get("file") == target_file:
                                local_seconds = max(0, target_seconds - int(fo.get("start_offset", 0)))
                                break
                        break
            elif paths:
                audio_path = Path(paths[0])

        if not audio_path or not audio_path.exists():
            messagebox.showinfo(
                "Open audio",
                f"Couldn't locate the audio file for this anchor.\n"
                f"Timestamp: {anchor.get('timestamp')}\n"
                f"File: {anchor.get('file')}\n\n"
                f"The session metadata may be missing audio paths."
            )
            return

        # Open at timestamp. Most desktop players support a #t=seconds URL fragment
        # via VLC; otherwise we just open the file and copy the timestamp to clipboard.
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(anchor.get("timestamp", ""))
        except Exception:
            pass

        opened = False
        # Try VLC if available
        for vlc_cmd in (["vlc", f"--start-time={int(local_seconds)}", str(audio_path)],):
            try:
                subprocess.Popen(vlc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                opened = True
                break
            except FileNotFoundError:
                pass

        if not opened:
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", str(audio_path)])
                elif os.name == "nt":
                    os.startfile(str(audio_path))  # type: ignore
                else:
                    subprocess.run(["xdg-open", str(audio_path)])
            except Exception as e:
                messagebox.showerror("Open audio", f"Could not open: {e}")
                return

        messagebox.showinfo(
            "Audio opened",
            f"Opened {audio_path.name}\n\nSeek to: {anchor.get('timestamp')}\n"
            f"(Timestamp copied to clipboard. If your player didn't auto-seek, paste it.)"
        )

    # ============= SPEAKERS TAB =============
    def _build_speakers_tab(self):
        f = self.speakers_frame
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Label(top, text="Campaign:").pack(side="left")
        self.sp_campaign_var = tk.StringVar(value=self.config.default_campaign)
        ttk.Entry(top, textvariable=self.sp_campaign_var, width=30).pack(side="left", padx=6)
        ttk.Button(top, text="Load Registry", command=self._load_speaker_registry).pack(side="left")
        ttk.Button(top, text="Save", command=self._save_speaker_registry).pack(side="left", padx=4)
        ttk.Button(top, text="Remove Selected", command=self._remove_speaker).pack(side="left", padx=4)

        ttk.Label(f, text="Known voice prints. Double-click a name to rename. "
                  "Voices are recognised automatically across sessions.",
                  foreground="#888").pack(padx=10, anchor="w")

        list_frame = ttk.Frame(f)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        cols = ("name", "sessions", "samples")
        self.sp_tree = ttk.Treeview(list_frame, columns=cols, show="headings")
        for c, w in [("name", 300), ("sessions", 300), ("samples", 100)]:
            self.sp_tree.heading(c, text=c.title())
            self.sp_tree.column(c, width=w, anchor="w")
        self.sp_tree.pack(fill="both", expand=True)
        self.sp_tree.bind("<Double-1>", self._on_speaker_dblclick)

        self._current_registry: Optional[SpeakerRegistry] = None

    def _load_speaker_registry(self):
        name = self.sp_campaign_var.get().strip() or "default"
        path = Path(self.config.campaign_root) / name / "speaker_registry.json"
        self._current_registry = SpeakerRegistry(path)
        self._refresh_speaker_tree()

    def _refresh_speaker_tree(self):
        for i in self.sp_tree.get_children():
            self.sp_tree.delete(i)
        if self._current_registry is None:
            return
        for sp_name, info in self._current_registry.data.get("speakers", {}).items():
            sessions = ", ".join(str(s) for s in info.get("sessions", []))
            samples = info.get("sample_count", 1)
            self.sp_tree.insert("", "end", values=(sp_name, sessions, samples))

    def _on_speaker_dblclick(self, event):
        sel = self.sp_tree.selection()
        if not sel or self._current_registry is None:
            return
        item = self.sp_tree.item(sel[0])
        old_name = item["values"][0]
        new_name = _ask_string(self.root, "Rename", f"New name for '{old_name}':", initial=old_name)
        if new_name and new_name != old_name:
            self._current_registry.rename(old_name, new_name)
            self._refresh_speaker_tree()

    def _remove_speaker(self):
        sel = self.sp_tree.selection()
        if not sel or self._current_registry is None:
            return
        name = self.sp_tree.item(sel[0])["values"][0]
        if messagebox.askyesno("Remove", f"Remove voice print for '{name}'?"):
            self._current_registry.remove(name)
            self._refresh_speaker_tree()

    def _save_speaker_registry(self):
        if self._current_registry is None:
            return
        try:
            self._current_registry.save()
            messagebox.showinfo("Saved", "Speaker registry saved.")
        except Exception as e:
            messagebox.showerror("Save", f"Failed: {e}")

    # ============= SETTINGS TAB =============
    def _build_settings_tab(self):
        f = self.settings_frame
        canvas = tk.Canvas(f, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        pad = {"padx": 10, "pady": 4}
        row = 0

        def header(text):
            nonlocal row
            ttk.Label(inner, text=text, font=("", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(12, 2)
            )
            row += 1

        def field(label, var, width=60, show=None):
            nonlocal row
            ttk.Label(inner, text=label).grid(row=row, column=0, sticky="e", **pad)
            ttk.Entry(inner, textvariable=var, width=width, show=show or "").grid(
                row=row, column=1, sticky="we", **pad
            )
            row += 1

        header("OpenAI / OpenAI-compatible")
        ttk.Label(inner, text="(works with OpenRouter, LM Studio, vLLM, Groq, Together — change Base URL)",
                  foreground="#888").grid(row=row, column=0, columnspan=2, sticky="w", padx=10)
        row += 1
        self.openai_key_var = tk.StringVar(value=self.config.openai_api_key)
        field("API Key:", self.openai_key_var, show="*")
        self.openai_model_var = tk.StringVar(value=self.config.openai_model)
        field("Model:", self.openai_model_var)
        self.openai_base_var = tk.StringVar(value=self.config.openai_base_url)
        field("Base URL:", self.openai_base_var)

        header("Anthropic")
        self.anth_key_var = tk.StringVar(value=self.config.anthropic_api_key)
        field("API Key:", self.anth_key_var, show="*")
        self.anth_model_var = tk.StringVar(value=self.config.anthropic_model)
        field("Model:", self.anth_model_var)

        header("Ollama (local)")
        self.ollama_base_var = tk.StringVar(value=self.config.ollama_base_url)
        field("Base URL:", self.ollama_base_var)
        self.ollama_model_var = tk.StringVar(value=self.config.ollama_model)
        field("Model:", self.ollama_model_var)

        header("Speaker Diarization (pyannote.audio)")
        ttk.Label(inner, text="Get a free token at https://hf.co/settings/tokens and accept the model terms at "
                  "https://hf.co/pyannote/speaker-diarization-3.1 and https://hf.co/pyannote/embedding",
                  foreground="#888", wraplength=700).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=10
        )
        row += 1
        self.hf_token_var = tk.StringVar(value=self.config.hf_token)
        field("HF Token:", self.hf_token_var, show="*")
        self.diar_device_var = tk.StringVar(value=self.config.diarization_device)
        ttk.Label(inner, text="Diarization Device:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Combobox(inner, textvariable=self.diar_device_var,
                     values=["cpu", "cuda"], state="readonly", width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1
        self.match_thresh_var = tk.DoubleVar(value=self.config.speaker_match_threshold)
        ttk.Label(inner, text="Match threshold (0.5–0.9):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(inner, textvariable=self.match_thresh_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        header("LLM tuning")
        self.temp_var = tk.DoubleVar(value=self.config.llm_temperature)
        ttk.Label(inner, text="Temperature:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(inner, textvariable=self.temp_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        row += 1
        self.maxtok_var = tk.IntVar(value=self.config.llm_max_tokens)
        ttk.Label(inner, text="Max tokens:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(inner, textvariable=self.maxtok_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Button(inner, text="Save Settings", command=self._save_settings).grid(
            row=row, column=1, sticky="w", padx=10, pady=12
        )
        row += 1
        ttk.Label(inner, text=f"Config file: {CONFIG_PATH}", foreground="#888").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 12)
        )

        inner.columnconfigure(1, weight=1)

    def _save_settings(self):
        self.config.openai_api_key = self.openai_key_var.get().strip()
        self.config.openai_model = self.openai_model_var.get().strip()
        self.config.openai_base_url = self.openai_base_var.get().strip()
        self.config.anthropic_api_key = self.anth_key_var.get().strip()
        self.config.anthropic_model = self.anth_model_var.get().strip()
        self.config.ollama_base_url = self.ollama_base_var.get().strip()
        self.config.ollama_model = self.ollama_model_var.get().strip()
        self.config.hf_token = self.hf_token_var.get().strip()
        self.config.diarization_device = self.diar_device_var.get().strip()
        try:
            self.config.speaker_match_threshold = float(self.match_thresh_var.get())
            self.config.llm_temperature = float(self.temp_var.get())
            self.config.llm_max_tokens = int(self.maxtok_var.get())
        except Exception:
            pass
        try:
            self.config.save()
            messagebox.showinfo("Settings", f"Saved to {CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("Settings", f"Failed: {e}")

    # ============= FILE / PROCESS HELPERS =============
    def add_files(self):
        filetypes = [
            ("Audio files", " ".join(f"*{e}" for e in AUDIO_EXTS)),
            ("All files", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)
        for p in paths:
            path = Path(p)
            if path not in self.selected_files:
                self.selected_files.append(path)
                self.file_tree.insert("", "end", values=(path.name, _format_duration(path)))

    def clear_files(self):
        self.selected_files.clear()
        for i in self.file_tree.get_children():
            self.file_tree.delete(i)
        self._set_status("Idle.")
        self.progress["value"] = 0

    def _move_file(self, delta: int):
        sel = self.file_tree.selection()
        if not sel:
            return
        idx = self.file_tree.index(sel[0])
        new_idx = idx + delta
        if 0 <= new_idx < len(self.selected_files):
            self.selected_files[idx], self.selected_files[new_idx] = (
                self.selected_files[new_idx], self.selected_files[idx]
            )
            self.file_tree.move(sel[0], "", new_idx)

    def choose_output(self):
        d = filedialog.askdirectory(title="Choose campaigns root folder",
                                     initialdir=self.output_var.get())
        if d:
            self.output_var.set(d)

    def open_output(self):
        folder = self.last_output_folder
        if not folder:
            folder = Path(self.output_var.get()) / self.campaign_var.get()
        folder = Path(folder)
        if not folder.exists():
            messagebox.showinfo("Output", f"Folder does not exist yet:\n{folder}")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            elif os.name == "nt":
                os.startfile(str(folder))  # type: ignore
            else:
                subprocess.run(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror("Open folder", f"Could not open folder:\n{e}")

    def start_processing(self):
        if self.processing:
            return
        if not self.selected_files:
            messagebox.showwarning("No files", "Please add at least one audio file.")
            return
        campaign_name = self.campaign_var.get().strip() or "default"
        self.config.whisper_model = self.whisper_var.get()
        self.config.whisper_device = self.device_var.get()
        self.config.llm_backend = self.llm_var.get()
        self.config.campaign_root = self.output_var.get()
        self.config.default_campaign = campaign_name
        self.config.enable_diarization = self.diarize_var.get()

        if self.diarize_var.get() and not self.config.hf_token:
            if not messagebox.askyesno(
                "No HF token",
                "Diarization is enabled but no HF token is set in Settings.\n\n"
                "It will fail and be skipped. Continue anyway?"
            ):
                return

        self.processing = True
        self.start_btn.configure(state="disabled")
        self.progress["value"] = 0
        self._set_status("Starting...")
        self._log(f"=== Processing session for campaign '{campaign_name}' ===")

        threading.Thread(
            target=self._run_pipeline,
            args=(campaign_name, list(self.selected_files),
                  self.journal_var.get(), self.memory_var.get(),
                  self.wiki_var.get(), self.diarize_var.get(),
                  self.vector_var.get(), self.anchor_var.get()),
            daemon=True,
        ).start()

    def _run_pipeline(self, campaign_name, files, journal, memory, wiki, diarize, vector, anchors):
        try:
            pipeline = SessionPipeline(
                self.config,
                campaign_name=campaign_name,
                logger=self._log,
                progress=self._progress,
            )
            self.last_pipeline = pipeline
            result = pipeline.process(
                files,
                generate_journal=journal,
                update_memory=memory,
                export_wiki_flag=wiki,
                enable_diarization=diarize,
                enable_vector=vector,
                enable_anchors=anchors,
            )
            self.last_output_folder = Path(result["session_folder"])
            self.last_unknown_speakers = result.get("unknown_speakers", [])
            self._log(f"✓ Done. Output: {result['session_folder']}")
            self._set_status(f"Finished: {result['title']}")
            if self.last_unknown_speakers:
                self.root.after(100, self._show_speaker_dialog)
        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}")
            self._log(traceback.format_exc())
            self._set_status(f"Error: {e}")
        finally:
            self.processing = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))

    def _show_speaker_dialog(self):
        if not self.last_unknown_speakers or self.last_pipeline is None:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Name New Speakers")
        dlg.geometry("520x420")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text=f"{len(self.last_unknown_speakers)} new voice(s) detected. "
                 f"Give each one a name (or leave blank to keep SPEAKER_X). "
                 f"Names you provide will be remembered for future sessions.",
            wraplength=480, foreground="#444", justify="left",
        ).pack(padx=12, pady=(12, 6))

        entries: Dict[str, tk.StringVar] = {}
        body = ttk.Frame(dlg)
        body.pack(fill="both", expand=True, padx=12, pady=6)
        for sp in self.last_unknown_speakers:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=sp["label"], width=14).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var, width=30).pack(side="left", padx=4)
            ttk.Label(row, text=f"(from {sp.get('sample_file','')})",
                      foreground="#888").pack(side="left")
            entries[sp["label"]] = var

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=(8, 12))

        def commit():
            assignments = {label: var.get().strip() for label, var in entries.items()}
            try:
                self.last_pipeline.commit_speaker_names(self.last_output_folder, assignments)
                self._log(f"[speakers] Committed: { {k: v for k, v in assignments.items() if v} }")
            except Exception as e:
                messagebox.showerror("Commit", f"Failed: {e}")
                return
            dlg.destroy()

        ttk.Button(btns, text="Save Names", command=commit).pack(side="right")
        ttk.Button(btns, text="Skip", command=dlg.destroy).pack(side="right", padx=6)

    # ============= LOGGING =============
    def _log(self, msg: str):
        self.log_queue.put(("log", msg))

    def _set_status(self, msg: str):
        self.log_queue.put(("status", msg))

    def _progress(self, pct: float, msg: str):
        self.log_queue.put(("progress", pct, msg))

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "status":
                    self.status_var.set(item[1])
                elif kind == "progress":
                    self.progress["value"] = max(0, min(100, item[1] * 100))
                    self.status_var.set(item[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _append_log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def _ask_string(parent, title, prompt, initial=""):
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent)
    dlg.grab_set()
    ttk.Label(dlg, text=prompt).pack(padx=12, pady=(12, 4))
    var = tk.StringVar(value=initial)
    entry = ttk.Entry(dlg, textvariable=var, width=30)
    entry.pack(padx=12, pady=4)
    entry.focus_set()
    result = {"value": None}
    def ok():
        result["value"] = var.get()
        dlg.destroy()
    def cancel():
        dlg.destroy()
    btns = ttk.Frame(dlg)
    btns.pack(pady=(4, 12))
    ttk.Button(btns, text="OK", command=ok).pack(side="left", padx=4)
    ttk.Button(btns, text="Cancel", command=cancel).pack(side="left", padx=4)
    entry.bind("<Return>", lambda e: ok())
    parent.wait_window(dlg)
    return result["value"]


def run():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()
