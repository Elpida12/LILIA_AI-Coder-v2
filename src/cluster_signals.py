"""Cluster lifecycle signaling between the AI coder and cluster_ctl.py."""

import json
import os
import time
from pathlib import Path

DRAIN_FILE = "cluster_draining"
CHECKPOINT_FILE = "resume_checkpoint.json"
PID_FILE = "ai_coder.pid"


def _flag_path(workspace: str | Path, name: str) -> Path:
    p = Path(workspace) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_drain_flag(workspace: str | Path, reason: str = "user-requested") -> None:
    """Signal that the cluster should drain (finish current work then stop)."""
    path = _flag_path(workspace, DRAIN_FILE)
    data = {"reason": reason, "timestamp": time.time()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def check_drain_requested(workspace: str | Path) -> tuple[bool, str]:
    """Return (draining, reason) if a drain has been requested."""
    path = _flag_path(workspace, DRAIN_FILE)
    if not path.exists():
        return False, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return True, data.get("reason", "unknown")
    except (json.JSONDecodeError, OSError):
        return True, "unknown (corrupt drain flag)"


def clear_drain_flag(workspace: str | Path) -> None:
    _flag_path(workspace, DRAIN_FILE).unlink(missing_ok=True)


def save_checkpoint(workspace: str | Path, data: dict) -> Path:
    path = _flag_path(workspace, CHECKPOINT_FILE)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_checkpoint(workspace: str | Path) -> dict | None:
    path = _flag_path(workspace, CHECKPOINT_FILE)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def remove_checkpoint(workspace: str | Path) -> None:
    _flag_path(workspace, CHECKPOINT_FILE).unlink(missing_ok=True)


def write_pid_file(workspace: str | Path, pid: int | None = None) -> Path:
    if pid is None:
        pid = os.getpid()
    path = _flag_path(workspace, PID_FILE)
    path.write_text(str(pid), encoding="utf-8")
    return path


def read_pid_file(workspace: str | Path) -> int | None:
    path = _flag_path(workspace, PID_FILE)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


class DrainRequested(Exception):
    """Raised when the cluster signals a drain and the current work should stop."""
    pass
