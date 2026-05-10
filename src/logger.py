"""Structured JSONL + styled terminal logger (Rich-powered).

All JSONL disk logging stays the same. Terminal output is upgraded via
RichOutput for colors, italics, panels, and markdown rendering.
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from src.rich_output import RichOutput


class Logger:
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    def __init__(self, logs_dir: str, level: str = "INFO", log_to_terminal: bool = True,
                 show_llm_output: bool = False):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.level = self.LEVELS.get(level.upper(), 1)
        self.log_to_terminal = log_to_terminal
        self.show_llm_output = show_llm_output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.logs_dir / f"session_{ts}.jsonl"
        self.session_log.touch()

        # Rich styling wrapper (gracefully degrades if Rich unavailable)
        self.rich = RichOutput(enabled=log_to_terminal)

    def _write(self, event_type: str, **data: Any):
        entry = {"timestamp": datetime.now().isoformat(), "event": event_type, **data}
        with open(self.session_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def _should_log(self, level: str) -> bool:
        return self.LEVELS.get(level.upper(), 1) >= self.level

    def debug(self, msg: str, **data: Any):
        if self._should_log("DEBUG"):
            self._write("debug", message=msg, **data)
            if self.log_to_terminal:
                self.rich.log("DEBUG", msg)

    def info(self, msg: str, **data: Any):
        if self._should_log("INFO"):
            self._write("info", message=msg, **data)
            if self.log_to_terminal:
                self.rich.log("INFO", msg)

    def warning(self, msg: str, **data: Any):
        if self._should_log("WARNING"):
            self._write("warning", message=msg, **data)
            if self.log_to_terminal:
                self.rich.log("WARNING", msg)

    def error(self, msg: str, exception: Exception | None = None, **data: Any):
        if self._should_log("ERROR"):
            entry = {"message": msg, **data}
            if exception:
                entry["exception"] = str(exception)
                entry["traceback"] = traceback.format_exc()
            self._write("error", **entry)
            if self.log_to_terminal:
                self.rich.log("ERROR", msg)

    def llm_call(self, agent: str, messages: list, response: str, thinking: str = "",
                 tokens_prompt: int = 0, tokens_completion: int = 0, duration: float = 0.0,
                 finish_reason: str = "", streamed: bool = False,
                 tool_calls: list[dict] | None = None):
        # Always write FULL content to the JSONL log file
        self._write(
            "llm_call",
            agent=agent,
            message_count=len(messages),
            response=response,
            thinking=thinking,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            duration_seconds=round(duration, 2),
            finish_reason=finish_reason,
        )
        # Print styled summary line
        if self.log_to_terminal:
            self.rich.llm_summary(agent, tokens_prompt, tokens_completion,
                                   duration, finish_reason)

        # Optionally print the full LLM output live to the terminal
        if self.show_llm_output and self.log_to_terminal:
            self.rich.llm_reasoning(thinking)
            self.rich.llm_response(response, tool_calls)

    def tool_call(self, agent: str, tool_name: str, arguments: dict, result: str, success: bool = True):
        self._write(
            "tool_call",
            agent=agent,
            tool=tool_name,
            arguments=arguments,
            result_preview=result[:500],
            success=success,
        )
        if self.log_to_terminal:
            self.rich.tool_call(agent, tool_name, success, result[:1500])

    def rule(self, title: str, char: str = "="):
        self._write("rule", title=title)
        if self.log_to_terminal:
            self.rich.rule(title, char)

    def summary(self, result: dict):
        """Print a styled project summary."""
        if self.log_to_terminal:
            self.rich.summary(result)
