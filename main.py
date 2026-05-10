#!/usr/bin/env python3
"""new_AI_CODER — local multi-agent AI programmer."""

import asyncio
import sys
import yaml
from pathlib import Path

from src.coordinator import Coordinator
from src.llm_backend import LLMError
from src.cluster_signals import load_checkpoint, remove_checkpoint


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "config.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def main():
    config = load_config()
    project_name = None
    resume_checkpoint = None

    # Auto-detect checkpoint for resume
    workspace_root = Path(config["workspace"]["root"]).resolve()
    checkpoint = load_checkpoint(workspace_root)
    if checkpoint:
        print(f"Found checkpoint from {checkpoint.get('timestamp', 'unknown')} — resuming")
        resume_checkpoint = checkpoint

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if "--help" in args or "-h" in args:
            print("""
Usage: python main.py [OPTIONS] [PROMPT]

  AI Coder — local multi-agent AI programmer and system assistant.
  Run with a prompt directly, or interactively if no prompt given.

Options:
  --name NAME       Project name (uses existing folder if present)
  --mode MODE       Override mode: "coding" or "assistant"
  --help, -h        Show this help message

Modes:
  coding     Full multi-agent pipeline: DESIGN -> IMPLEMENT -> VERIFY
             Best for building software projects from scratch.
  assistant  Single-task ReAct mode with sysadmin tools.
             Best for troubleshooting, system tasks, downloads, etc.

Examples:
  python main.py "Build a snake game"
  python main.py "Fix auth bug" --name my_project
  python main.py "Check disk usage" --mode assistant
""")
            sys.exit(0)
        if "--name" in args:
            idx = args.index("--name")
            if idx + 1 < len(args):
                project_name = args[idx + 1]
                del args[idx:idx + 2]
        if "--mode" in args:
            idx = args.index("--mode")
            if idx + 1 < len(args):
                mode_override = args[idx + 1]
                del args[idx:idx + 2]
                if mode_override in ("coding", "assistant"):
                    config.setdefault("features", {})
                    config["features"]["mode"] = mode_override
                    print(f"Mode override: {mode_override}")
                else:
                    print(f"Warning: unknown mode '{mode_override}', expected 'coding' or 'assistant'")
        user_prompt = " ".join(args)
    else:
        print("\nDescribe your project (empty line to submit):")
        lines = []
        while True:
            line = input()
            if lines and line == "":
                break
            if line == "":
                continue
            lines.append(line)
        user_prompt = "\n".join(lines).strip()
        if not user_prompt:
            print("No prompt. Exiting.")
            sys.exit(1)
        try:
            project_name = input("Project name (Enter to auto-generate): ").strip() or None
        except EOFError:
            pass

    coord = Coordinator(config)

    try:
        result = await coord.run(
            user_prompt,
            project_name=project_name,
            resume_checkpoint=resume_checkpoint,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except LLMError as exc:
        print(f"\nLLM Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFatal: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        await coord.llm.close()

    # Clear checkpoint on success (so old projects don't auto-resume)
    if result.get("status") == "complete":
        remove_checkpoint(workspace_root)

    # Print styled summary
    coord.logger.summary(result)


if __name__ == "__main__":
    asyncio.run(main())
