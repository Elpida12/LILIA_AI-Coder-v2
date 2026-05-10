"""Tool definitions, schemas, and dispatch."""

import inspect
from dataclasses import dataclass


# OpenAI-format tool definitions
_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read contents of a file relative to project root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit an existing file by replacing old_text with new_text. Use exact text from read_file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories in the project.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "default": "."}
            },
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_project",
        "description": "Search for text across all project files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_memory",
        "description": "Update a section of shared project memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "content": {"type": "object"},
            },
            "required": ["section", "content"],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal task completion with a verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "verdict": {"type": "string", "enum": ["pass", "issues", "fail", "complete"]},
                "details": {"type": "object"},
            },
            "required": ["summary", "verdict"],
        },
    },
    {
        "name": "browse_web",
        "description": "Fetch the text content of a web page. Returns page body as markdown/plain text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"},
                "max_chars": {"type": "integer", "default": 8000, "description": "Maximum characters to return"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "download_file",
        "description": "Download a file from a URL to a local path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to download"},
                "path": {"type": "string", "description": "Local relative path to save the file"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["url", "path"],
        },
    },
    {
        "name": "read_system_info",
        "description": "Read system information: OS, kernel, architecture, CPU, memory, disk usage.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]

_ROLE_TOOLS = {
    "architect": frozenset({"read_file", "list_files", "search_project", "update_memory", "task_complete"}),
    "implementer": frozenset({"read_file", "write_file", "edit_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
    "verifier": frozenset({"read_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
    "repair": frozenset({"read_file", "write_file", "edit_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
    "sysadmin": frozenset({"read_file", "write_file", "edit_file", "list_files", "search_project", "run_command", "update_memory", "task_complete", "browse_web", "download_file", "read_system_info"}),
}


class ToolRegistry:
    def __init__(self, fs, shell, git, memory, logger):
        self.fs = fs
        self.shell = shell
        self.git = git
        self.memory = memory
        self.logger = logger

    def get_schemas(self, role: str | None = None) -> list[dict]:
        allowed = _ROLE_TOOLS.get(role)
        tools = _TOOL_DEFINITIONS if allowed is None else [
            t for t in _TOOL_DEFINITIONS if t["name"] in allowed
        ]
        return [{"type": "function", "function": t} for t in tools]

    def execute(self, name: str, arguments: dict, agent_name: str = "unknown") -> str:
        allowed = _ROLE_TOOLS.get(agent_name)
        if allowed is not None and name not in allowed:
            return f"Error: '{agent_name}' cannot use '{name}'. Allowed: {sorted(allowed)}"
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            sig = inspect.signature(handler)
            valid = {k: v for k, v in arguments.items() if k in sig.parameters}
            result = handler(**valid)
            self.logger.tool_call(agent_name, name, arguments, result, success=True)
            return str(result)
        except Exception as exc:
            msg = f"Error: {exc}"
            self.logger.tool_call(agent_name, name, arguments, msg, success=False)
            return msg

    def _tool_read_file(self, path: str) -> str:
        return self.fs.read(path)

    def _tool_write_file(self, path: str, content: str) -> str:
        return self.fs.write(path, content)

    def _tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        return self.fs.edit(path, old_text, new_text)

    def _tool_list_files(self, directory: str = ".") -> str:
        return self.fs.list_files(directory)

    def _tool_run_command(self, command: str, timeout: int = 120) -> str:
        # Flush staged writes so shell commands can see current files
        self.fs.commit()
        return self.shell.run(command, timeout)

    def _tool_search_project(self, query: str) -> str:
        return self.fs.search_project(query)

    def _tool_update_memory(self, section: str, content: dict) -> str:
        result = self.memory.update_section(section, content)
        # Verify persistence (read-back check)
        if not self.memory._path.exists():
            return f"Error: Memory file not found after update at {self.memory._path}"
        return result

    def _tool_task_complete(self, summary: str, verdict: str = "complete", details: dict | None = None) -> str:
        return f"TASK_COMPLETE [{verdict}]: {summary}"

    # ------------------------------------------------------------------
    # New sysadmin tools
    # ------------------------------------------------------------------
    def _tool_browse_web(self, url: str, max_chars: int = 8000) -> str:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                # Try to decode as text
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("utf-8", errors="replace")
                # Very rough HTML stripping
                import re
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]*>", "", text)
                text = re.sub(r"\n\s*\n+", "\n\n", text)
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[TRUNCATED: {len(text) - max_chars} chars omitted]"
                return text
        except urllib.error.URLError as exc:
            return f"Error fetching {url}: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_download_file(self, url: str, path: str, timeout: int = 120) -> str:
        import urllib.request
        import urllib.error
        try:
            target = self.fs._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                target.write_bytes(resp.read())
            return f"Downloaded {url} -> {path} ({target.stat().st_size} bytes)"
        except urllib.error.URLError as exc:
            return f"Error downloading {url}: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_read_system_info(self) -> str:
        import platform
        import shutil
        lines = [
            f"OS: {platform.system()} {platform.release()}",
            f"Architecture: {platform.machine()}",
            f"Processor: {platform.processor()}",
            f"Python: {platform.python_version()}",
            f"Node: {platform.node()}",
        ]
        try:
            mem = shutil.disk_usage("/")
            lines.append(f"Root disk: total={mem.total//(1024**3)}GB, free={mem.free//(1024**3)}GB, used={mem.used//(1024**3)}GB")
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                lines.append(f"Memory:\n{out.stdout.strip()}")
        except Exception:
            pass
        try:
            out = subprocess.run(["uname", "-a"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                lines.append(f"Kernel: {out.stdout.strip()}")
        except Exception:
            pass
        return "\n".join(lines)
