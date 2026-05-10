"""Shell command execution within project directory."""

import os
import re
import select
import signal
import subprocess
import sys
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# pty is Unix-only; import guarded for cross-platform use
try:
    import pty
except ImportError:  # pragma: no cover
    pty = None  # type: ignore

# Lines matching these patterns are dropped from command output before the
# LLM sees them.  This prevents `.venv`, `__pycache__`, `.git`, etc. from
# exploding the context window when the agent runs broad commands like
# `find . -name "*.py"`.
PROJECT_IGNORE_PATTERNS = [
    r"\.venv[\\/]",
    r"__pycache__[\\/]",
    r"\.git[\\/]",
    r"node_modules[\\/]",
    r"\.pytest_cache[\\/]",
    r"\.mypy_cache[\\/]",
    r"\.tox[\\/]",
]

# Hard safety cap for command output.  If a command produces more chars
# than this *after* filtering, we cut at the last complete line and add
# a clear notice.  This prevents a single legitimate large file or log
# from killing the context budget.
MAX_OUTPUT_CHARS = 15_000

# Patterns that ALWAYS require user confirmation before execution.
# These are checked against the resolved command string.
DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"rm\s+-rf\s+/\*"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\>\s*/etc/"),
    re.compile(r"\>\s*/boot/"),
    re.compile(r"curl\s+.*\|\s*sh"),
    re.compile(r"wget\s+.*\|\s*sh"),
    re.compile(r"sudo\s+rm"),
    re.compile(r"chown\s+-R\s+/"),
    re.compile(r"chmod\s+-R\s+/"),
    re.compile(r"fdisk\s+/dev/"),
    re.compile(r"mkfs\.[a-z]+\s+/dev/"),
    re.compile(r"\bpkexec\b"),
    re.compile(r"\bsu\s+-\s*root"),
]

# Patterns that are considered "safe" and never need confirmation.
# If the command contains ONLY safe operations, skip the gate.
_SAFE_BARE_COMMANDS = {
    "ls", "cat", "find", "grep", "ps", "df", "du",
    "journalctl", "dmesg", "ip", "ping", "hostname",
    "uname", "whoami", "pwd", "echo", "head", "tail",
    "less", "more", "wc", "sort", "uniq", "date", "uptime",
    "free", "top", "htop", "lsof", "netstat", "ss",
    "systemctl", "service", "apt-get", "apt", "dnf", "yum",
    "pacman", "snap", "flatpak", "brew",
    "curl", "wget", "git", "python", "python3", "pip",
    "npm", "yarn", "pnpm", "cargo", "go", "rustc",
}


def _prompt_confirm(command: str) -> bool:
    """Interactive terminal prompt for dangerous commands."""
    import sys
    print(f"\n[SECURITY] This command requires confirmation:")
    print(f"  {command}")
    print(f"Allow execution? (y/N): ", end="", flush=True)
    try:
        response = sys.stdin.readline().strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print(" — cancelled")
        return False


class ShellRunner:
    def __init__(self, cwd: Path, venv_path: Path | None = None, confirm_commands: bool = True):
        self.cwd = cwd.resolve()
        self.venv_path = venv_path.resolve() if venv_path else None
        self.confirm_commands = confirm_commands

    # --- Command gate ---
    def _needs_confirmation(self, command: str) -> bool:
        """Return True if command matches a dangerous pattern."""
        if not self.confirm_commands:
            return False
        for pat in DANGEROUS_PATTERNS:
            if pat.search(command):
                return True
        return False

    def _check_command(self, command: str) -> str | None:
        """Run the confirmation gate. Returns None if allowed, or an error string if blocked."""
        if not self._needs_confirmation(command):
            return None
        if not _prompt_confirm(command):
            return f"Command blocked by user: {command}"
        return None

    def _venv_python(self) -> str | None:
        """Resolve the venv Python executable path."""
        if not self.venv_path:
            return None
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        exe = "python.exe" if sys.platform == "win32" else "python"
        candidate = self.venv_path / bin_dir / exe
        return str(candidate) if candidate.exists() else None

    def _venv_pip(self) -> str | None:
        """Resolve the venv pip executable path."""
        if not self.venv_path:
            return None
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        candidate = self.venv_path / bin_dir / "pip"
        return str(candidate) if candidate.exists() else None

    def _resolve_command(self, command: str) -> str:
        """Replace bare python/python3/pip commands with venv equivalents.

        This ensures that agents always use the project's virtual environment,
        regardless of which Python binary name they choose. Handles:
        - 'python', 'python3' -> venv/bin/python
        - 'pip' -> venv/bin/pip
        - Commands in subshells (cd ... && python ...)
        - pip install/show commands
        
        Fix: Prevent double-prepending of venv paths by checking if command
        already contains a venv python path.
        
        Fix: Fall back to sys.executable if venv Python is not available,
        to avoid exit code 127 when venv setup fails or is incomplete.
        """
        venv_python = self._venv_python()
        
        # FALLBACK FIX: Use sys.executable if venv Python not found
        # This prevents exit code 127 (command not found) when venv is broken
        if not venv_python:
            venv_python = sys.executable

        venv_pip = self._venv_pip()

        # Fix: Check if command already contains a venv python path to avoid double-prepending
        if venv_python in command:
            # Command already uses venv python, don't modify it further
            return command

        # Replace bare python3 with venv python (must come before 'python')
        # Use negative lookbehind (?<!/) to avoid matching python3 in paths like /usr/bin/python3
        command = re.sub(r'(?<!/)python3(?![\w.-])', venv_python, command)
        # Replace bare python with venv python (but not python3, already handled)
        # Negative lookbehind for '3' to avoid matching what we just replaced
        # Also use (?<!/) to avoid matching python in paths like /path/python
        command = re.sub(r'(?<!/)python(?!3)(?![\w.-])', venv_python, command)

        # Replace bare pip with venv pip (but only if not already using venv pip)
        if venv_pip and venv_pip not in command:
            command = re.sub(r'\bpip\b', venv_pip, command)

        return command

    @staticmethod
    def _filter_output(text: str) -> str:
        """Strip lines that reference project noise directories.

        This is done *before* the LLM ever sees the output, so the LLM does
        not know data is missing — it simply sees a clean listing.
        """
        if not text:
            return text
        lines = text.splitlines(keepends=True)
        filtered = []
        for line in lines:
            if any(re.search(pat, line) for pat in PROJECT_IGNORE_PATTERNS):
                continue
            filtered.append(line)
        return "".join(filtered)

    @staticmethod
    def _enforce_max_length(text: str) -> str:
        """Cut text at last complete line if it exceeds MAX_OUTPUT_CHARS.

        Adds a clear boundary notice so the LLM knows it is truncated
        (unlike mid-string truncation which confuses the LLM).
        """
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        # Cut at last complete line before the limit
        cut = text.rfind("\n", 0, MAX_OUTPUT_CHARS)
        if cut == -1:
            cut = MAX_OUTPUT_CHARS
        return (
            text[:cut]
            + f"\n\n[OUTPUT TRUNCATED: {len(text) - cut:,} characters omitted. "
            "Result was too large for context window.]\n"
        )

    @staticmethod
    def _inject_find_exclusions(command: str) -> str:
        """If the command is a broad `find` or `grep -r`, auto-inject exclusions.

        Only modifies commands that recurse without explicit path filters,
        so the LLM's intent is respected when it already narrowed the search.
        """
        cmd = command.strip()
        # Only touch find commands that do NOT already exclude .venv
        if cmd.startswith("find ") and "-not -path" not in cmd and "-prune" not in cmd:
            # Insert exclusions after the first word so they apply globally
            exclusions = " ".join(
                f"-not -path '*/{name}/*'" for name in (".venv", "__pycache__", ".git", "node_modules")
            )
            # Insert right after 'find' (e.g. "find . -name '*.py'" -> "find . -not ... -name '*.py'")
            parts = cmd.split(None, 2)
            if len(parts) >= 2:
                return f"{parts[0]} {parts[1]} {exclusions} {parts[2] if len(parts) > 2 else ''}"
        return command

    def parse_exit_code(self, output: str) -> int:
        """Parse exit code from ShellRunner.run() output string.

        Returns the exit code if found, or -1 if unable to determine.
        """
        for line in output.split("\n"):
            if line.startswith("Exit code:"):
                try:
                    return int(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    return -1
        return -1

    def pty_run(self, command: str, timeout: int = 10) -> str:
        """Run a command inside a pseudo-terminal (PTY) and capture startup output.

        Useful for TTY/interactive apps (curses, blessed, etc.) that fail to
        initialize in a plain subprocess. The process is killed after `timeout`
        seconds, so this is intended as a smoke test — not a full run.

        On Windows (where `pty` is unavailable), falls back to a plain
        subprocess.run with a short timeout.
        """
        # --- Command gate: confirm dangerous operations ---
        gate_result = self._check_command(command)
        if gate_result:
            return gate_result

        if pty is None:
            # Windows / no PTY support — fall back to plain run
            return self.run(command, timeout=timeout)

        command = self._resolve_command(command)

        try:
            master_fd, slave_fd = pty.openpty()
        except OSError as exc:
            return f"PTY open failed: {exc}\nCommand: {command}\nExit code: -1"

        pid = os.fork()
        if pid == 0:
            # Child process: set up slave side of PTY
            os.close(master_fd)
            os.setsid()
            os.dup2(slave_fd, 0)  # stdin
            os.dup2(slave_fd, 1)  # stdout
            os.dup2(slave_fd, 2)  # stderr
            os.close(slave_fd)
            os.chdir(self.cwd)
            os.execl("/bin/sh", "sh", "-c", command)
            os._exit(127)

        # Parent process
        os.close(slave_fd)
        output_chunks: list[str] = []
        exit_code = -1

        try:
            ready, _, _ = select.select([master_fd], [], [], timeout)
            if ready:
                try:
                    data = os.read(master_fd, 8192)
                    if data:
                        output_chunks.append(data.decode(errors="replace"))
                except OSError as exc:
                    logger.debug(f"PTY read error: {exc}")

            # Gracefully terminate the child (ignore if already gone)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError as exc:
                logger.debug(f"SIGTERM on already-gone process: {exc}")

            # Reap the child and collect its exit code.
            # We do a blocking waitpid with a generous timeout to avoid zombies.
            # If the child is already gone, waitpid returns immediately.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    _, raw_status = os.waitpid(pid, os.WNOHANG)
                    if raw_status != 0:
                        if hasattr(os, "waitstatus_to_exitcode"):
                            exit_code = os.waitstatus_to_exitcode(raw_status)
                        else:
                            exit_code = raw_status // 256
                        break
                except (ChildProcessError, ProcessLookupError) as exc:
                    logger.debug(f"waitpid during reap: {exc}")
                    break
                time.sleep(0.05)
            else:
                # Child didn't exit after SIGTERM — force kill
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError as exc:
                    logger.debug(f"SIGKILL on already-gone process: {exc}")
                try:
                    _, raw_status = os.waitpid(pid, 0)
                    if hasattr(os, "waitstatus_to_exitcode"):
                        exit_code = os.waitstatus_to_exitcode(raw_status)
                    else:
                        exit_code = raw_status // 256
                except (ChildProcessError, ProcessLookupError) as exc:
                    logger.debug(f"waitpid after SIGKILL: {exc}")
        except Exception as exc:
            output_chunks.append(f"\nPTY error: {exc}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError as exc2:
                logger.debug(f"SIGKILL on already-gone process during cleanup: {exc2}")
            try:
                os.waitpid(pid, 0)
            except (ChildProcessError, ProcessLookupError) as exc2:
                logger.debug(f"waitpid during cleanup: {exc2}")
        finally:
            try:
                os.close(master_fd)
            except OSError as exc:
                logger.debug(f"close master_fd: {exc}")

        stdout = "".join(output_chunks)
        return (
            f"Command: {command}\n"
            f"Exit code: {exit_code}\n"
            f"STDOUT:\n{stdout}"
        )

    def run(self, command: str, timeout: int = 120) -> str:
        # Inject broad-recursion exclusions for known noisy commands
        command = self._inject_find_exclusions(command)
        command = self._resolve_command(command)

        # --- Command gate: confirm dangerous operations ---
        gate_result = self._check_command(command)
        if gate_result:
            return gate_result

        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.cwd,
                capture_output=True, text=True, timeout=timeout,
            )
            stdout = self._filter_output(proc.stdout or "")
            stderr = self._filter_output(proc.stderr or "")
            stdout = self._enforce_max_length(stdout)
            stderr = self._enforce_max_length(stderr)
            lines = [f"Command: {command}", f"Exit code: {proc.returncode}"]
            if stdout:
                lines.append(f"STDOUT:\n{stdout}")
            if stderr:
                lines.append(f"STDERR:\n{stderr}")
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as exc:
            return f"Command failed: {exc}"