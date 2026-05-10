"""Shared project memory with JSON persistence."""

import json
import copy
from datetime import datetime
from pathlib import Path


class ProjectMemory:
    def __init__(self, project_name: str, workspace_root: str):
        self.data = {
            "project": {
                "name": project_name,
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
            "design": {},
            "tasks": [],
            "task_history": [],
            "file_registry": {},
            "issues": [],
            "fixes": [],
            "decisions": [],
        }
        self._path = Path(workspace_root) / project_name / ".memory.json"

    def load(self) -> bool:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        self.data = loaded
                        return True
            except (json.JSONDecodeError, OSError) as exc:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load memory from {self._path}: {exc}")
        return False

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def get_compact(self, max_chars: int = 6000) -> str:
        compact = copy.deepcopy(self.data)
        text = json.dumps(compact, indent=2)
        if len(text) > max_chars:
            th = compact.get("task_history", [])
            if len(th) > 3:
                compact["task_history"] = [
                    {"prior_summary": f"... {len(th)-3} older tasks"}
                ] + th[-3:]
            issues = compact.get("issues", [])
            if len(issues) > 5:
                compact["issues"] = issues[-5:]
            # NEW: cap file_registry entries — they grow unbounded as tasks complete
            fr = compact.get("file_registry", {})
            if len(fr) > 20:
                items = list(fr.items())[-20:]
                compact["file_registry"] = dict(items)
                compact["file_registry"]["_note"] = f"... {len(fr)-20} earlier files omitted"
            # NEW: cap design/fixes/decisions depth
            design = compact.get("design", {})
            if len(design) > 30:
                items = list(design.items())[-30:]
                compact["design"] = dict(items)
                compact["design"]["_note"] = f"... {len(design)-30} earlier design items omitted"
            fixes = compact.get("fixes", [])
            if len(fixes) > 10:
                compact["fixes"] = fixes[-10:]
            decisions = compact.get("decisions", [])
            if len(decisions) > 10:
                compact["decisions"] = decisions[-10:]
            text = json.dumps(compact, indent=2)
        return text

    def update_section(self, section: str, content):
        if section not in self.data:
            return f"Error: unknown section '{section}'. Valid: {list(self.data.keys())}"
        if section == "project":
            if isinstance(content, dict):
                self.data["project"].update(content)
            else:
                return "Error: project section requires a dict"
        elif section in ("tasks", "task_history", "issues", "fixes", "decisions"):
            if isinstance(content, dict):
                self.data[section].append(content)
            elif isinstance(content, list):
                self.data[section].extend(content)
            else:
                return f"Error: {section} requires dict or list of dicts"
        elif section in ("file_registry", "design"):
            if isinstance(content, dict):
                self.data[section].update(content)
            else:
                return f"Error: {section} requires a dict"
        self.save()
        return f"Updated {section}."

    def add_task(self, task_id: str, description: str, status: str,
                 files_changed: list = None, summary: str = "",
                 duration: float = None, attempts: int = None):
        entry = {
            "task_id": task_id,
            "description": description,
            "status": status,
            "files_changed": files_changed or [],
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
        if duration is not None:
            entry["duration"] = round(duration, 2)
        if attempts is not None:
            entry["attempts"] = attempts
        self.data["task_history"].append(entry)
        self.save()
