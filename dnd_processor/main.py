"""Entry point for the D&D Session Processor.

GUI (default):   python main.py
CLI:             python main.py --cli --campaign mycampaign file1.mp3 file2.mp3
Ask a question:  python main.py --ask "When did we first meet Lord Brannis?" --campaign mycampaign
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from config import Config


def run_cli(args):
    from core.pipeline import SessionPipeline
    cfg = Config.load()
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.llm:
        cfg.llm_backend = args.llm
    if args.output:
        cfg.campaign_root = args.output

    def log(msg): print(msg)
    def progress(p, m): print(f"  [{p*100:5.1f}%] {m}")

    pipeline = SessionPipeline(cfg, campaign_name=args.campaign, logger=log, progress=progress)
    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            print(f"File not found: {f}", file=sys.stderr)
            sys.exit(1)

    result = pipeline.process(
        files,
        generate_journal=not args.no_journal,
        update_memory=not args.no_memory,
        export_wiki_flag=not args.no_wiki,
    )
    print(f"\nDone. Session folder: {result['session_folder']}")


def run_ask(args):
    from core.memory import CampaignMemory
    from core.llm import LLM
    from core.qa import CampaignQA

    cfg = Config.load()
    if args.llm:
        cfg.llm_backend = args.llm
    mem_path = Path(cfg.campaign_root) / args.campaign / "campaign_memory.json"
    if not mem_path.exists():
        print(f"No memory found at {mem_path}", file=sys.stderr)
        sys.exit(1)
    memory = CampaignMemory(mem_path)
    llm = LLM(cfg, logger=lambda m: None)
    qa = CampaignQA(llm, memory.data)
    print(qa.ask(args.ask))


def main():
    parser = argparse.ArgumentParser(description="D&D Session Processor")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")
    parser.add_argument("--campaign", default="default", help="Campaign name")
    parser.add_argument("--output", help="Campaigns root folder")
    parser.add_argument("--whisper-model", help="Override Whisper model (tiny/base/small/medium/large)")
    parser.add_argument("--llm", choices=["fallback", "openai", "ollama"], help="LLM backend")
    parser.add_argument("--no-journal", action="store_true", help="Skip journal rewrite")
    parser.add_argument("--no-memory", action="store_true", help="Skip campaign memory update")
    parser.add_argument("--no-wiki", action="store_true", help="Skip wiki export")
    parser.add_argument("--ask", help="Ask a question against a campaign's memory")
    parser.add_argument("files", nargs="*", help="Audio files (for --cli mode)")
    args = parser.parse_args()

    if args.ask:
        run_ask(args)
        return

    if args.cli:
        if not args.files:
            parser.error("--cli requires at least one audio file")
        run_cli(args)
        return

    # Default: GUI
    from gui.app import run
    run()


if __name__ == "__main__":
    main()
