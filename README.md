# AI Coder — Local Multi-Agent AI Programmer & System Assistant

A fully local, multi-agent AI coding system that designs, implements, verifies, and repairs software projects autonomously. Also doubles as a general system assistant for troubleshooting, downloads, and maintenance tasks.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20Compatible-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Coding Mode](#coding-mode)
  - [Assistant Mode](#assistant-mode)
- [Security](#security)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

---

## Overview

AI Coder orchestrates multiple specialized LLM agents through a state-machine coordinator to turn natural language prompts into working code. It runs entirely against your own local or remote OpenAI-compatible API (llama.cpp, Ollama, vLLM, etc.) — no cloud services required.

The system supports two operational modes:

| Mode | Purpose | Pipeline |
|------|---------|----------|
| **coding** | Build software from scratch | Architect → Implementer → Verifier → Repair |
| **assistant** | System tasks, troubleshooting, downloads | Single-task ReAct (Sysadmin agent) |

---

## Key Features

### Multi-Agent Pipeline
- **Architect Agent** — Decomposes user requests into a design document and dependency-aware task list
- **Implementer Agent** — Writes code for each task using native tool calling
- **Verifier Agent** — Runs syntax checks, import resolution, linting (Ruff, MyPy), and tests
- **Repair Agent** — Fixes bugs found by Verifier with targeted patches
- **Sysadmin Agent** — General system assistant with web browsing, file download, and system info tools

### Execution Engine
- **DAG-based scheduling** — Tasks execute in topological order with parallelization of independent tasks
- **Automatic retry with escalation** — Failed tasks retry up to 5x; repeated errors escalate to RepairAgent
- **Git checkpointing** — Every phase is committed; rollbacks restore clean state on failure
- **Auto virtualenv** — Each project gets its own `.venv` with pytest, ruff, mypy pre-installed
- **Checkpoint / Resume** — Crashed or interrupted runs resume from the last saved state

### Robustness
- **Context trimming** — Message history is automatically trimmed to stay within LLM context limits
- **Message flow validation** — Ensures tool-call ID ordering integrity after trimming
- **LLM response caching** — Disk + memory cache deduplicates identical requests
- **Circuit breaker** — Backs off automatically when the LLM API is unreachable
- **Truncation detection** — Detects truncated tool calls and requests shorter responses
- **Convergence detection** — Detects when an agent is stuck in an error loop and aborts

### Safety
- **Command confirmation gate** — Dangerous shell commands (`rm -rf /`, `mkfs`, `curl | sh`, etc.) require interactive user confirmation
- **Filesystem sandbox** — All file paths are resolved relative to the project root with escape prevention
- **Output filtering** — `.venv`, `__pycache__`, `.git` noise is stripped from command output before the LLM sees it
- **Output cap** — Shell output is hard-capped at 15,000 characters with clean truncation notices

### Terminal Experience
- **Rich-styled output** — Colored panels, badges, syntax highlighting, and markdown rendering
- **Graceful fallback** — Plain text output if Rich is unavailable
- **Structured JSONL logging** — Every LLM call, tool call, and phase transition is written to timestamped log files

---

## Architecture

```
User Prompt
    |
    v
+-----------+     +-------------+     +--------------+
| Architect | --> | Task DAG    | --> | Implementer  |
| (Design)  |     | (Validator) |     | (Code)       |
+-----------+     +-------------+     +--------------+
                                            |
                                            v
+-----------+     +-------------+     +--------------+
|  Repair   | <-- |  Verifier   | <-- |   Commit     |
| (Fix)     |     | (Test/Lint) |     |   + Git      |
+-----------+     +-------------+     +--------------+
                                            |
                                            v
                                    +--------------+
                                    |   Coordinator |
                                    |   (Summary)   |
                                    +--------------+
```

1. **Architect** produces a design + task list with dependencies
2. **DAGValidator** checks for cycles, missing dependencies, and test-before-source ordering
3. **Coordinator** executes the DAG in parallel batches (max 3 concurrent LLM calls)
4. Each task runs: Implement → Verify → (Repair if needed) → Commit
5. On success, a final smoke-test runs the project's entry point
6. On failure, the task rolls back to the last git checkpoint and retries

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai-coder

# Install dependencies
pip install -r requirements.txt

# Optional: install Rich for styled terminal output
pip install rich
```

### Requirements

- Python 3.11+
- An OpenAI-compatible LLM server running locally (llama.cpp, Ollama, vLLM, etc.)
- Git (for checkpointing)

---

## Configuration

Edit `config.yaml` to point at your LLM server:

```yaml
llm:
  base_url: "http://127.0.0.1:8080/v1"
  model: "minimax-m2.7"
  api_key: "no-key-needed"
  context_size: 60000
  reasoning_format: "deepseek"
  temperature: 1.0
  max_tokens: 32768
  cache_dir: "./logs/llm_cache"

agents:
  max_iterations: 100

workspace:
  root: "./workspace"
  logs_dir: "./logs"

logging:
  level: "DEBUG"
  log_to_terminal: true
  show_llm_output: true

features:
  mode: "coding"          # "coding" or "assistant"
  git_checkpoints: true
  venv_auto: true
  max_task_retries: 5
  stop_on_first_failure: true
  concurrent_llm_limit: 1

  assistant:
    confirm_commands: true
    danger_patterns:
      - "rm -rf /"
      - "mkfs"
      - "curl .*| *sh"
    safe_patterns:
      - "ls"
      - "cat"
      - "df"
```

### Key Settings

| Setting | Description |
|---------|-------------|
| `mode` | `coding` = full multi-agent pipeline; `assistant` = single-task sysadmin mode |
| `context_size` | Must match your model's actual context window |
| `reasoning_format` | `"deepseek"` enables reasoning_content / thinking tokens |
| `max_task_retries` | How many times a failed task retries before giving up |
| `stop_on_first_failure` | If `true`, aborts the entire project when any task fails |
| `concurrent_llm_limit` | Max parallel LLM calls (semaphore) |
| `confirm_commands` | If `true`, dangerous shell commands require user confirmation |

---

## Usage

### Coding Mode

Build software from a natural language prompt:

```bash
# Direct prompt
python main.py "Build a snake game in curses"

# Named project (uses existing folder if present)
python main.py "Build a REST API" --name my_api

# Interactive mode (multi-line prompt, then asks for project name)
python main.py
```

**Example session:**
```
Describe your project (empty line to submit):
> A CLI password manager with AES encryption
> and a TOTP generator

Project name (Enter to auto-generate): passman

========================================
  PROJECT: passman
========================================
[architect]  1200+3400 tok  4.2s  finish=stop
[implementer]  800+2100 tok  3.1s  finish=stop
[verifier]  600+800 tok  1.5s  finish=stop
...
========================================
  PROJECT COMPLETE
========================================
  Root : /home/user/ai-coder/workspace/passman
  Tasks: 5/5
```

### Assistant Mode

Use the sysadmin agent for system tasks without triggering the full coding pipeline:

```bash
python main.py "Check disk usage on my system" --mode assistant
python main.py "Download the latest release of neovim" --mode assistant
python main.py "Fix the nginx config so it serves static files" --mode assistant
```

---

## Security

### Command Confirmation Gate

When `confirm_commands: true` (default), any shell command matching dangerous patterns triggers an interactive terminal prompt:

```
[SECURITY] This command requires confirmation:
  rm -rf /tmp/old_project
Allow execution? (y/N):
```

Dangerous patterns include: `rm -rf /`, `mkfs`, `dd if=`, `> /etc/`, `curl | sh`, `sudo rm`, `chown -R /`, etc.

### Filesystem Sandbox

All file operations are resolved relative to the project root. Absolute paths and directory-escape attempts (e.g., `../../etc/passwd`) are rejected with a `ValueError`.

### Staged Writes

File writes go through an in-memory overlay (`FileSandbox`) and are only persisted when `commit()` is called. Failed tasks trigger `revert()`, discarding uncommitted changes.

---

## Project Structure

```
ai-coder/
├── main.py                     # Entry point / CLI
├── config.yaml                 # User configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── workspace/                  # Generated projects live here
│   └── <project_name>/
│       ├── .venv/              # Auto-created virtualenv
│       ├── .git/               # Checkpoint history
│       └── .memory.json        # Project memory (design, tasks, history)
├── logs/                       # Session logs & LLM cache
│   ├── session_*.jsonl         # Structured event logs
│   └── llm_cache/              # Disk-cached LLM responses
└── src/
    ├── coordinator.py          # State machine + DAG executor
    ├── llm_backend.py          # OpenAI-compatible client (async, streaming, cache)
    ├── agent_base.py           # ReAct loop, context trimming, flow validation
    ├── logger.py               # JSONL + Rich terminal output
    ├── rich_output.py          # Terminal styling engine
    ├── memory.py               # Shared project memory (JSON persistence)
    ├── dag_validator.py        # Cycle detection + topological scheduling
    ├── cluster_signals.py      # Checkpoint/resume + drain signaling
    ├── agents/
    │   ├── architect.py        # Design + task decomposition
    │   ├── implementer.py      # Code writing
    │   ├── verifier.py         # Pre-checks + test execution
    │   ├── repair.py           # Bug fixing
    │   └── sysadmin.py         # General system assistant
    ├── tools/
    │   ├── registry.py         # Tool schemas + dispatch
    │   ├── fs.py               # Sandboxed filesystem (read/write/edit/list/search)
    │   ├── shell.py            # Shell execution (with gate, filtering, PTY)
    │   └── git.py              # Git checkpointing (init, commit, restore)
    └── prompts/
        ├── architect.txt       # System prompt for design agent
        ├── implementer.txt     # System prompt for code agent
        ├── verifier.txt        # System prompt for test agent
        ├── repair.txt          # System prompt for fix agent
        └── sysadmin.txt        # System prompt for assistant agent
```

---

## Advanced Features

### Resume from Checkpoint

If a run crashes, is interrupted, or the cluster drains, the coordinator saves a checkpoint to `workspace/resume_checkpoint.json`. The next run auto-detects it and resumes from where it left off — completed tasks are skipped, failed tasks retry.

### Context Window Management

The base agent (`AgentBase`) implements aggressive but safe context trimming:
- Always keeps the system prompt and first user message
- Drops oldest middle messages first, preserving recent context
- Estimates tokens conservatively (~4 chars/token) with runtime calibration
- Validates message flow after trimming to prevent "tool role without matching tool_call" API errors
- Hard cap at 50 messages with a 20% response reserve

### LLM Caching

Identical prompts (same messages, tools, temperature, model, thinking_budget) are served from an in-memory LRU cache backed by disk JSON files. This dramatically speeds up retries and repeated verification runs.

### Circuit Breaker

After 5 consecutive LLM failures (connection errors, HTTP 4xx/5xx), the backend opens a 60-second circuit breaker to avoid hammering a downed server.

### PTY Smoke Tests

For TTY-based applications (curses, blessed, Rich Live), the coordinator runs a PTY smoke test instead of a plain subprocess to catch initialization crashes that would otherwise appear as silent failures.

---

## Troubleshooting

### "LLM unreachable after 3 attempts"
- Check that your LLM server is running at the `base_url` in `config.yaml`
- Verify the server supports OpenAI-compatible `/v1/chat/completions`
- If using `reasoning_format: "deepseek"`, ensure the server supports the `thinking` parameter

### "Invalid task graph"
- The Architect produced tasks with circular dependencies or missing task IDs
- Rerun with the same prompt; the system will retry with a fresh design

### "Context severely over budget"
- Your model's context window is too small for the project
- Reduce `context_size` in config to match your model's actual limit
- Lower `max_tokens` to leave more room for the prompt

### Rich styling not appearing
- Install Rich: `pip install rich`
- The system falls back to plain text automatically if Rich is missing

---

## Requirements

- **Python**: 3.11 or newer
- **httpx**: >=0.27 (async HTTP client)
- **pyyaml**: YAML configuration parsing
- **pydantic**: >=2.0 (optional validation for Architect output)
- **rich**: Optional, for styled terminal output
- **git**: Required for checkpoint/rollback functionality
- **LLM Server**: Any OpenAI-compatible API (llama.cpp, Ollama, vLLM, TabbyAPI, etc.)

---

## License

MIT License — see the project root for full license text.

---

*Built with patience, automated testing, and a lot of retries.*
