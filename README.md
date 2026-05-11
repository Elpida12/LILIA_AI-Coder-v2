<p align="center">LILIA</p>
<p align="center">Local Intent-Driven Language Implementation Architecture</p>

> A fully-local, multi-agent coding assistant that turns a natural-language
> project description into a complete, runnable Python codebase - without
> sending a single token to the cloud.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20Compatible-green)


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
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Overview

Lilia orchestrates multiple specialized LLM agents through a state-machine coordinator to turn natural language prompts into working code. It runs entirely against your own local or remote OpenAI-compatible API - no cloud services required.

The system supports two operational modes:

| Mode | Purpose | Pipeline |
|------|---------|----------|
| **coding** | Build software from scratch | Architect - Implementer - Verifier - Repair |
| **assistant** | System tasks, troubleshooting, downloads | Single-task ReAct (Sysadmin agent) |

---

## Key Features

### Multi-Agent Pipeline
- **Architect Agent** - Decomposes user requests into a design document and dependency-aware task list
- **Implementer Agent** - Writes code for each task using native tool calling
- **Verifier Agent** - Runs syntax checks, import resolution, linting (Ruff, MyPy), and tests
- **Repair Agent** - Fixes bugs found by Verifier with targeted patches
- **Sysadmin Agent** - General system assistant with web browsing, file download, and system info tools

### Execution Engine
- **DAG-based scheduling** - Tasks execute in topological order with parallelization of independent tasks
- **Automatic retry with escalation** - Failed tasks retry up to 5x; repeated errors escalate to RepairAgent
- **Git checkpointing** - Every phase is committed; rollbacks restore clean state on failure
- **Auto virtualenv** - Each project gets its own `.venv` with pytest, ruff, mypy pre-installed
- **Checkpoint / Resume** - Crashed or interrupted runs resume from the last saved state

### Robustness
- **Context trimming** - Message history is automatically trimmed to stay within LLM context limits
- **Message flow validation** - Ensures tool-call ID ordering integrity after trimming
- **LLM response caching** - Disk + memory cache deduplicates identical requests
- **Circuit breaker** - Backs off automatically when the LLM API is unreachable
- **Truncation detection** - Detects truncated tool calls and requests shorter responses
- **Convergence detection** - Detects when an agent is stuck in an error loop and aborts

### Safety
- **Command confirmation gate** - Dangerous shell commands (`rm -rf /`, `mkfs`, `curl | sh`, etc.) require interactive user confirmation
- **Filesystem sandbox** - All file paths are resolved relative to the project root with escape prevention
- **Output filtering** - `.venv`, `__pycache__`, `.git` noise is stripped from command output before the LLM sees it
- **Output cap** - Shell output is hard-capped at 15,000 characters with clean truncation notices

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

### Assistant Mode

Use the sysadmin agent for system tasks without triggering the full coding pipeline:

```bash
python main.py "Check disk usage on my system" --mode assistant
python main.py "Download the latest release of neovim" --mode assistant
python main.py "Fix the nginx config so it serves static files" --mode assistant
```

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

## Requirements

- **Python**: 3.11 or newer
- **httpx**: >=0.27 (async HTTP client)
- **pyyaml**: YAML configuration parsing
- **pydantic**: >=2.0 (optional validation for Architect output)
- **rich**: Optional, for styled terminal output
- **git**: Required for checkpoint/rollback functionality
- **LLM Server**: Any OpenAI-compatible API (llama.cpp, Ollama, vLLM, TabbyAPI, etc.)

---
