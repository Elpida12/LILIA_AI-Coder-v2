"""Rich-based terminal styling for AI Coder.

Provides a styled Console wrapper and reusable formatters for
LLM reasoning, responses, tool calls, and phase headers.
"""

import sys
from datetime import datetime
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import box
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


class RichOutput:
    """Styled terminal output with graceful fallback to plain text."""

    # Color/style definitions
    STYLES = {
        "timestamp": "dim",
        "debug": "dim bright_black",
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "rule_title": "bold bright_cyan",
        "rule_border": "cyan",
        "reasoning": "italic dim magenta",
        "response": "bright_white",
        "tool_name": "bold green",
        "tool_result": "green",
        "agent_badge": "bold bright_blue on grey15",
        "phase_panel": "bright_cyan",
        "summary_ok": "bold green",
        "summary_fail": "bold red",
        "summary_warn": "bold yellow",
    }

    def __init__(self, enabled: bool = True, width: int | None = None):
        self.enabled = enabled and _HAS_RICH
        if self.enabled:
            self.console = Console(
                width=width,
                stderr=False,
                highlight=False,
            )
            self.console_stderr = Console(
                width=width,
                stderr=True,
                highlight=False,
            )
        else:
            self.console = None
            self.console_stderr = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _plain(self, level: str, msg: str, file: Any = sys.stdout):
        ts = self._ts()
        print(f"[{ts}] [{level}] {msg}", file=file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log(self, level: str, msg: str, **kwargs: Any):
        """Emit a log line with level-based styling."""
        if not self.enabled:
            stream = sys.stderr if level == "ERROR" else sys.stdout
            self._plain(level, msg, file=stream)
            return

        ts = Text(f"[{self._ts()}] ", style=self.STYLES["timestamp"])
        badge = Text(f"[{level}] ", style=self.STYLES.get(level.lower(), "white"))
        body = Text(msg)
        line = Text.assemble(ts, badge, body)

        target = self.console_stderr if level == "ERROR" else self.console
        target.print(line, **kwargs)

    def rule(self, title: str, char: str = "="):
        """Print a phase/rule separator."""
        if not self.enabled:
            print(f"\n{char * 60}\n  {title}\n{char * 60}")
            return

        # Use a panel for phase headers
        panel = Panel(
            Text(title, style=self.STYLES["rule_title"]),
            border_style=self.STYLES["rule_border"],
            box=box.ROUNDED,
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def llm_summary(self, agent: str, prompt_tok: int, comp_tok: int,
                    duration: float, finish_reason: str):
        """Compact one-line LLM call summary."""
        if not self.enabled:
            print(
                f"LLM [{agent}] {prompt_tok}+{comp_tok} tok, "
                f"{duration:.1f}s, finish={finish_reason}"
            )
            return

        badge = Text(f" {agent} ", style=self.STYLES["agent_badge"])
        meta = Text(
            f"  {prompt_tok}+{comp_tok} tok  {duration:.1f}s  finish={finish_reason}",
            style="dim",
        )
        self.console.print(Text.assemble(badge, meta))

    def llm_reasoning(self, thinking: str):
        """Print LLM reasoning block in italics with a muted color."""
        if not thinking or not thinking.strip():
            return
        if not self.enabled:
            print(f"Reasoning: {thinking}")
            return

        label = Text("Reasoning\n", style="bold dim magenta")
        body = Text(thinking, style=self.STYLES["reasoning"])
        self.console.print(label)
        self.console.print(body)
        self.console.print()

    def llm_response(self, response: str, tool_calls: list[dict] | None = None):
        """Print LLM response or tool-call summary."""
        if not self.enabled:
            if tool_calls:
                names = [tc.get("name", "unknown") for tc in tool_calls]
                print(f"Response: <tool_calls: {', '.join(names)}>")
            elif response.strip():
                print(f"Response: {response}")
            else:
                print("Response: (empty content)")
            return

        if tool_calls:
            names = [tc.get("name", "unknown") for tc in tool_calls]
            self.console.print(
                Text.assemble(
                    Text("Tool calls: ", style="bold"),
                    Text(", ").join(Text(n, style=self.STYLES["tool_name"]) for n in names),
                )
            )
        elif response.strip():
            # Try to render markdown if it looks like it contains formatting
            if any(marker in response for marker in ("**", "##", "```", "- ")):
                self.console.print(Markdown(response))
            else:
                self.console.print(Text(response, style=self.STYLES["response"]))
        else:
            self.console.print(Text("(empty content)", style="dim"))
        self.console.print()

    def tool_call(self, agent: str, tool_name: str, success: bool = True,
                  result_preview: str = ""):
        """Print a compact tool-call result line."""
        if not self.enabled:
            status = "OK" if success else "FAIL"
            print(f"Tool [{agent}] {tool_name}() -> {status}")
            return

        badge = Text(f" {agent} ", style=self.STYLES["agent_badge"])
        name = Text(f" {tool_name}() ", style=self.STYLES["tool_name"])
        status_text = Text(" OK ", style="bold white on green" if success else "bold white on red")
        self.console.print(Text.assemble(badge, name, status_text))
        if result_preview:
            preview = result_preview[:1500]
            self.console.print(Text(preview, style="dim"))
            self.console.print()

    def summary(self, result: dict):
        """Print a styled project summary table/panel."""
        if not self.enabled:
            self._plain_summary(result)
            return

        status = result.get("status", "unknown")
        if status == "complete":
            panel_title = "PROJECT COMPLETE"
            border_style = self.STYLES["summary_ok"]
        elif status == "partial":
            panel_title = "PROJECT PARTIAL FAILURE"
            border_style = self.STYLES["summary_warn"]
        else:
            panel_title = f"PROJECT {status.upper()}"
            border_style = self.STYLES["summary_fail"]

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("key", style="bold cyan", no_wrap=True)
        table.add_column("value", style="bright_white")

        table.add_row("Root", result.get("project_root", "N/A"))
        table.add_row("Tasks", f"{result.get('tasks_completed', 0)}/{result.get('tasks_total', 0)}")
        if result.get("phase"):
            table.add_row("Phase", result["phase"])
        if result.get("error"):
            table.add_row("Error", result["error"])
        if result.get("failed_tasks"):
            table.add_row("Failed", ", ".join(result["failed_tasks"]))

        panel = Panel(
            table,
            title=panel_title,
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def _plain_summary(self, result: dict):
        print("\n" + "=" * 60)
        if result.get("status") == "complete":
            print("  PROJECT COMPLETE")
            print(f"  Root : {result.get('project_root')}")
            print(f"  Tasks: {result.get('tasks_completed', 0)}/{result.get('tasks_total', 0)}")
        else:
            print(f"  PROJECT {result.get('status', 'unknown').upper()}")
            if result.get("phase"):
                print(f"  Phase: {result['phase']}")
            if result.get("error"):
                print(f"  Error: {result['error']}")
            if result.get("failed_tasks"):
                print(f"  Failed tasks: {', '.join(result['failed_tasks'])}")
        print("=" * 60)

    def code_block(self, code: str, language: str = "python"):
        """Render a syntax-highlighted code block if possible."""
        if not self.enabled:
            print(f"```{language}\n{code}\n```")
            return
        self.console.print(Syntax(code, language, theme="monokai", line_numbers=False))

    def markdown(self, text: str):
        """Render markdown text."""
        if not self.enabled:
            print(text)
            return
        self.console.print(Markdown(text))
