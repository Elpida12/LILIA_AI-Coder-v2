"""State-machine coordinator — async with DAG parallel execution."""

import asyncio
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.llm_backend import LLMBackend
from src.memory import ProjectMemory
from src.logger import Logger
from src.tools.fs import FileSandbox
from src.tools.shell import ShellRunner
from src.tools.git import GitCheckpoint
from src.tools.registry import ToolRegistry
from src.agents import ArchitectAgent, ImplementerAgent, VerifierAgent, RepairAgent, SysadminAgent
from src.dag_validator import DAGValidator
from src.cluster_signals import DrainRequested


class TaskDAG:
    """Manages dependency-aware task ordering."""

    def __init__(self, tasks: list[dict]):
        self.tasks = {t["id"]: t for t in tasks}
        self.completed: set[str] = set()
        self.failed: set[str] = set()

    def ready(self) -> list[dict]:
        """Return tasks ready to run, sorted: source tasks before test tasks."""
        ready_tasks = [
            t for tid, t in self.tasks.items()
            if tid not in self.completed | self.failed
            and all(dep in self.completed for dep in t.get("dependencies", []))
        ]
        # Sort: non-test tasks first
        def _is_test(t: dict) -> bool:
            tid = t.get("id", "").lower()
            desc = t.get("description", "").lower()
            targets = [f.lower() for f in t.get("target_files", [])]
            return "test" in tid or "test" in desc or any("test" in tf for tf in targets)
        ready_tasks.sort(key=lambda t: (1 if _is_test(t) else 0, t.get("id", "")))
        return ready_tasks

    def mark(self, tid: str, ok: bool):
        (self.completed if ok else self.failed).add(tid)

    def all_done(self) -> bool:
        return self.completed | self.failed == set(self.tasks)


class Coordinator:
    """
    Async state machine for project generation:
        IDLE -> DESIGN   (Architect produces design + task list)
        DESIGN -> [for each task or batch of independent tasks]
            IMPLEMENT -> VERIFY
        -> DONE
    """

    SEMAPHORE_VALUE = 3  # Limit concurrent LLM calls

    def __init__(self, config: dict):
        self.cfg = config
        self.logger = Logger(
            logs_dir=config["workspace"]["logs_dir"],
            level=config["logging"]["level"],
            log_to_terminal=config["logging"]["log_to_terminal"],
            show_llm_output=config["logging"].get("show_llm_output", False),
        )

        llm_cfg = config["llm"]
        self.llm = LLMBackend(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["model"],
            api_key=llm_cfg.get("api_key", ""),
            context_size=llm_cfg["context_size"],
            reasoning_format=llm_cfg["reasoning_format"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            logger=self.logger,
        )

        self.workspace = Path(config["workspace"]["root"]).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.agents_cfg = config.get("agents", {})
        self.features = config.get("features", {})
        self.prompts_dir = str(Path(__file__).parent / "prompts")
        self._semaphore = asyncio.Semaphore(self.features.get("concurrent_llm_limit", self.SEMAPHORE_VALUE))

    def _normalize_task_dependencies(self, tasks: list[dict]) -> list[dict]:
        """Fix LLM hallucinations where dependencies list filenames instead of task IDs."""
        # Build a mapping from target_files -> task_id
        file_to_tid = {}
        for t in tasks:
            tid = t.get("id")
            if not tid:
                continue
            for f in t.get("target_files", []):
                file_to_tid[f] = tid

        all_tids = {t.get("id") for t in tasks if t.get("id")}

        fixed = []
        for t in tasks:
            deps = t.get("dependencies", [])
            new_deps = []
            for dep in deps:
                if dep in all_tids:
                    new_deps.append(dep)
                elif dep in file_to_tid:
                    resolved = file_to_tid[dep]
                    self.logger.warning(
                        f"[coordinator] Resolving dependency '{dep}' -> '{resolved}' "
                        f"for task {t.get('id')}"
                    )
                    new_deps.append(resolved)
                else:
                    new_deps.append(dep)
            t["dependencies"] = new_deps
            fixed.append(t)
        return fixed

    def _save_task_checkpoint(self, project_name: str, user_prompt: str,
                              tasks: list[dict], dag: TaskDAG | None = None,
                              phase: str = "implementation"):
        data = {
            "version": 1,
            "project_name": project_name,
            "user_prompt": user_prompt,
            "tasks": tasks,
            "phase": phase,
            "dag_completed": list(dag.completed) if dag else [],
            "dag_failed": list(dag.failed) if dag else [],
            "timestamp": time.time(),
        }
        from src.cluster_signals import save_checkpoint
        path = save_checkpoint(self.workspace, data)
        self.logger.info(f"Checkpoint saved: {path}")

    def _drain_requested(self) -> bool:
        from src.cluster_signals import check_drain_requested
        draining, reason = check_drain_requested(self.workspace)
        if draining:
            self.logger.info(f"Drain requested: {reason}")
        return draining

    def _clear_checkpoint(self):
        from src.cluster_signals import remove_checkpoint
        remove_checkpoint(self.workspace)

    async def run(self, user_prompt: str | None = None, project_name: str | None = None,
                  resume_checkpoint: dict | None = None) -> dict:
        is_resume = resume_checkpoint is not None

        if is_resume:
            project_name = resume_checkpoint.get("project_name") or project_name
            user_prompt = resume_checkpoint.get("user_prompt") or user_prompt

        if not project_name:
            words = (user_prompt or "").split()[:4]
            project_name = "_".join(w.lower() for w in words if w.isalnum())
            if not project_name:
                project_name = f"project_{int(time.time())}"
        else:
            if "/" in project_name or "\\" in project_name or ".." in project_name:
                raise ValueError(f"Unsafe project_name: {project_name!r}")

        project_root = self.workspace / project_name
        project_root.mkdir(parents=True, exist_ok=True)

        self.logger.rule(f"PROJECT: {project_name}", char="=")
        self.logger.info(f"Root: {project_root}")
        self.logger.info(f"Prompt: {user_prompt}")
        if is_resume:
            self.logger.info("Resuming from checkpoint")

        # --- memory ---
        memory = ProjectMemory(project_name, str(self.workspace))
        was_loaded = memory.load()
        memory.update_section("project", {"name": project_name, "description": user_prompt})
        # Fix: Reset stale memory from previous runs so architect designs fresh tasks.
        # Skip reset when resuming — we need the existing design and task history.
        if was_loaded and not is_resume:
            self.logger.info("Resetting stale memory from previous run")
            memory.data["tasks"] = []
            memory.data["task_history"] = []
            memory.data["design"] = {}
            memory.save()

        # --- mode detection ---
        mode = self.features.get("mode", "coding")
        is_assistant = mode == "assistant"

        # --- venv ---
        venv_path = project_root / ".venv"
        if self.features.get("venv_auto", True) and not is_assistant:
            await self._setup_venv(venv_path)

        # --- tools ---
        fs = FileSandbox(project_root)
        confirm = self.features.get("assistant", {}).get("confirm_commands", True)
        shell = ShellRunner(project_root, venv_path, confirm_commands=confirm)
        git = GitCheckpoint(project_root)
        if not is_assistant:
            git.init()
        tools = ToolRegistry(fs, shell, git, memory, self.logger)

        # --- ASSISTANT MODE: single-task ReAct, skip pipeline ---
        if is_assistant:
            self.logger.rule("ASSISTANT MODE", char="-")
            result = await self._run_sysadmin(tools, memory, user_prompt, project_root)
            if result["status"] == "complete":
                memory.add_task(
                    task_id="assistant",
                    description=user_prompt or "",
                    status="complete",
                    files_changed=result.get("files_changed", []),
                    summary=result["result"],
                )
                return {
                    "status": "complete",
                    "project_root": str(project_root),
                    "tasks_total": 1,
                    "tasks_completed": 1,
                    "task_history": memory.data.get("task_history", []),
                }
            else:
                return {
                    "status": result["status"],
                    "project_root": str(project_root),
                    "error": result.get("result", "unknown error"),
                    "task_history": memory.data.get("task_history", []),
                }

        # --- PHASE 1: DESIGN ---
        self.logger.rule("PHASE 1: DESIGN", char="-")
        if is_resume:
            tasks = resume_checkpoint.get("tasks", [])
            self.logger.info(f"Resuming with {len(tasks)} checkpointed tasks")
        else:
            tasks = await self._run_design(tools, memory, user_prompt)
        if not tasks:
            return self._fail("design", "No tasks produced by architect", project_root)

        # --- VALIDATE TASK GRAPH ---
        tasks = self._normalize_task_dependencies(tasks)
        validator = DAGValidator(tasks)
        error = validator.validate_or_fail()
        if error:
            self.logger.error(f"Task graph validation failed: {error}")
            return self._fail("design", f"Invalid task graph: {error}", project_root)
        ordered_tasks = validator.schedule_source_first()
        self.logger.info(f"Task schedule order: {ordered_tasks}")

        memory.update_section("tasks", tasks)
        git.checkpoint("design complete")
        fs.commit()  # commit architect writes

        # --- PHASE 2: EXECUTE TASKS (DAG parallel) ---
        self.logger.rule("PHASE 2: IMPLEMENTATION", char="-")
        dag = TaskDAG(tasks)
        failed_tasks: list[str] = []

        # Restore DAG state from checkpoint if resuming
        if is_resume:
            for tid in resume_checkpoint.get("dag_completed", []):
                dag.mark(tid, True)
            for tid in resume_checkpoint.get("dag_failed", []):
                dag.mark(tid, False)
                failed_tasks.append(tid)
            self.logger.info(
                f"Restored DAG state: {len(dag.completed)} completed, {len(dag.failed)} failed"
            )

        # Track retry error patterns for escalation
        retry_errors: dict[str, str] = {}

        stop_early = False
        with ThreadPoolExecutor(max_workers=4) as pool:
            loop = asyncio.get_running_loop()
            while not dag.all_done() and not stop_early:
                batch = dag.ready()
                if not batch:
                    break
                coros = [
                    self._run_task_cycle_semaphore(tid, task, tools, memory, git, project_root, pool, retry_errors)
                    for task in batch
                    if (tid := task["id"])
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for task, res in zip(batch, results):
                    tid = task["id"]
                    if isinstance(res, Exception):
                        self.logger.error(f"[{tid}] Exception: {res}")
                        dag.mark(tid, False)
                        failed_tasks.append(tid)
                    elif res:
                        dag.mark(tid, True)
                    else:
                        dag.mark(tid, False)
                        failed_tasks.append(tid)
                        if self.features.get("stop_on_first_failure", False):
                            # Mark all remaining tasks in this batch as failed
                            for remaining_task in batch:
                                rid = remaining_task["id"]
                                if rid != tid and rid not in dag.completed and rid not in dag.failed:
                                    dag.mark(rid, False)
                                    failed_tasks.append(rid)
                            stop_early = True
                            break

                # Checkpoint after each batch so resume works granularly
                self._save_task_checkpoint(project_name, user_prompt, tasks, dag, phase="implementation")

                # Check if a cluster drain was requested
                if self._drain_requested():
                    self.logger.info("Drain requested — checkpointing and raising DrainRequested")
                    raise DrainRequested("Cluster drain requested by cluster_ctl")

        # --- PHASE 3: FINAL CHECK ---
        self.logger.rule("PHASE 3: FINAL CHECK", char="-")
        entry = memory.data.get("project", {}).get("entry_point", "")

        def _is_test_task_count(tid: str) -> bool:
            """Consistent test-task heuristic for counting (same as TaskDAG/DAGValidator)."""
            task_info = dag.tasks.get(tid, {})
            low_id = tid.lower()
            desc = task_info.get("description", "").lower()
            targets = [f.lower() for f in task_info.get("target_files", [])]
            return "test" in low_id or "test" in desc or any("test" in tf for tf in targets)

        source_tasks_attempted = len([t for t in tasks if not _is_test_task_count(t["id"])])
        source_tasks_completed = len([
            tid for tid in dag.completed
            if not _is_test_task_count(tid)
        ])

        if entry and (project_root / entry).exists():
            # Fix: Use different timeouts for different application types
            # GUI applications like VPython need longer startup times
            import os
            is_gui_app = False
            is_tty_app = False

            # Check if it's a GUI application by looking for common GUI libraries
            try:
                with open(project_root / entry, 'r') as f:
                    content = f.read().lower()
                    gui_indicators = ['vpython', 'tkinter', 'pygame', 'pyqt', 'pyside',
                                    'matplotlib', 'plotly', 'dash', 'streamlit']
                    tty_indicators = ['curses', 'ncurses', 'blessed', 'rich.live', 'getch']
                    is_gui_app = any(indicator in content for indicator in gui_indicators)
                    is_tty_app = any(indicator in content for indicator in tty_indicators)
            except (OSError, IOError) as exc:
                self.logger.warning(f"Could not read entry file for GUI/TTY check: {exc}")

            if is_tty_app:
                # TTY apps need a PTY to initialize; smoke-test for startup crashes
                proc = shell.pty_run(f"python {entry}", timeout=5)
                self.logger.info(f"Entry module PTY smoke test (TTY app):\n{proc[:1000]}")
                exit_code = shell.parse_exit_code(proc)

                # Determine if the app crashed during startup.
                # PTY smoke tests are killed by SIGTERM after timeout, so a negative
                # exit code is normal for both healthy and crashing apps. We look at
                # captured output for traceback / error signatures instead.
                has_traceback = "Traceback" in proc or "traceback" in proc
                has_exception = any(
                    kw in proc
                    for kw in ("_curses.error", "AttributeError", "TypeError",
                               "NameError", "ImportError", "ValueError",
                               "RuntimeError", "OSError", "Exception",
                               "An error occurred", "no attribute", "Error:")
                )
                crashed = (
                    (exit_code is not None and exit_code > 0)  # definite non-zero exit
                    or has_traceback
                    or has_exception
                )

                if crashed:
                    self.logger.error(
                        f"Entry module '{entry}' crashed during PTY smoke test.\n"
                        f"Exit code: {exit_code}\n"
                        f"Captured output:\n{proc}"
                    )
                    failed_tasks.append("_entry_point")
                else:
                    self.logger.info(
                        f"Entry module '{entry}' PTY smoke test passed "
                        f"(exit code {exit_code}, no startup crash detected)."
                    )
            else:
                # Set timeout based on application type
                timeout = 60 if is_gui_app else 30
                proc = shell.run(f"python {entry}", timeout=timeout)
                self.logger.info(f"Entry module check (timeout: {timeout}s):\n{proc[:500]}")
                # Fix #3: Parse and check exit code
                exit_code = shell.parse_exit_code(proc)
                # Exit code -1 means the process was killed by timeout (e.g. interactive
                # CLI apps that wait on input()). Only treat positive exit codes as
                # genuine failures — a timeout on an interactive app is expected.
                if exit_code is not None and exit_code > 0:
                    self.logger.error(
                        f"Entry module '{entry}' failed with exit code {exit_code}"
                    )
                    failed_tasks.append("_entry_point")
                elif exit_code == -1:
                    self.logger.info(
                        f"Entry module '{entry}' timed out after {timeout}s "
                        "(expected for interactive CLI app)."
                    )
        elif entry:
            self.logger.error(f"Entry point '{entry}' is missing.")

        # If no source tasks were attempted/completed and only tests exist, treat as failure
        if source_tasks_attempted == 0 and len(tasks) > 0:
            self.logger.error("No source-file tasks were attempted. Only test tasks were scheduled.")
            self.logger.rule("PROJECT PARTIAL FAILURE", char="!")
            return {
                "status": "failed",
                "project_root": str(project_root),
                "tasks_total": len(tasks),
                "tasks_completed": len(dag.completed),
                "failed_tasks": failed_tasks,
                "task_history": memory.data.get("task_history", []),
            }

        # --- SUMMARY ---
        total = len(tasks)
        completed = len(dag.completed)
        if failed_tasks:
            self.logger.rule("PROJECT PARTIAL FAILURE", char="!")
            return {
                "status": "partial",
                "project_root": str(project_root),
                "tasks_total": total,
                "tasks_completed": completed,
                "failed_tasks": failed_tasks,
                "task_history": memory.data.get("task_history", []),
            }

        self.logger.rule("PROJECT COMPLETE", char="=")
        return {
            "status": "complete",
            "project_root": str(project_root),
            "tasks_total": total,
            "tasks_completed": completed,
            "task_history": memory.data.get("task_history", []),
        }

    async def _run_design(self, tools: ToolRegistry, memory: ProjectMemory, prompt: str) -> list[dict]:
        agent = ArchitectAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        result = await agent.run({"description": prompt, "type": "design"})
        if result["status"] != "complete":
            self.logger.error(f"Architect failed: {result['result']}")
            return []

        design = result.get("design", {})
        if design:
            memory.update_section("design", design)
            if "entry_point" in design:
                memory.update_section("project", {"entry_point": design["entry_point"]})

        tasks = result.get("tasks", [])
        if not tasks:
            self.logger.error("Architect produced no tasks")
            return []
        return tasks

    async def _run_sysadmin(self, tools: ToolRegistry, memory: ProjectMemory, prompt: str | None, project_root: Path) -> dict:
        """Single-task ReAct for assistant mode — no DAG, no verify, no git."""
        task = {
            "description": prompt or "",
            "type": "sysadmin",
            "project_root": str(project_root),
            "workspace_hint": (
                f"Project workspace is at: {project_root}\n"
                f"All file paths are relative to this directory.\n"
                f"You can also use absolute paths for system-level tasks."
            ),
        }
        agent = SysadminAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        return await agent.run(task)

    async def _run_task_cycle_semaphore(self, tid: str, task: dict, tools: ToolRegistry,
                                         memory: ProjectMemory, git: GitCheckpoint,
                                         project_root: Path, pool: ThreadPoolExecutor,
                                         retry_errors: dict[str, str] | None = None) -> bool:
        async with self._semaphore:
            return await self._run_task_cycle(tid, task, tools, memory, git, project_root, pool, retry_errors)

    async def _run_task_cycle(self, tid: str, task: dict, tools: ToolRegistry,
                               memory: ProjectMemory, git: GitCheckpoint,
                               project_root: Path, pool: ThreadPoolExecutor,
                               retry_errors: dict[str, str] | None = None) -> bool:
        max_retries = self.features.get("max_task_retries", 5)
        task_start = time.time()
        files_changed: list[str] = []
        attempts = 0
        prev_error_signature = ""
        retry_errors = retry_errors or {}

        # Use a copy of the task dict to avoid mutating shared references
        # across concurrent task cycles and retry iterations.
        task = {**task}

        for attempt in range(1, max_retries + 1):
            attempts += 1
            lessons_learned = ""

            if attempt > 1:
                self.logger.info(f"[{tid}] Retry {attempt}/{max_retries}")

                # Fix #6: Differentiate between truncation/exhaustion failures and
                # logic failures. For truncation, preserve partial work.
                # Fix #3B: Also preserve work on LLM infrastructure failures
                # (message history corruption, 400 errors) — these are scaffolding
                # bugs, not LLM logic errors. The partial code is still valid.
                prev_err = retry_errors.get(tid, "")
                is_truncation_failure = (
                    "truncation_loop" in prev_err
                    or ("Hit" in prev_err and "iterations" in prev_err)
                )
                is_llm_infra_failure = (
                    "LLM failure" in prev_err
                    or "LLM at" in prev_err
                    or "unreachable" in prev_err
                    or "Message has tool role" in prev_err
                    or "400 Bad Request" in prev_err
                )

                if is_truncation_failure or is_llm_infra_failure:
                    # Preserve staged changes — partial work is valid
                    tools.fs.commit()
                    git.checkpoint(f"retry {tid} attempt {attempt} (partial work preserved)")
                    if is_llm_infra_failure:
                        self.logger.info(f"[{tid}] Preserving partial work from LLM infrastructure failure")
                    else:
                        self.logger.info(f"[{tid}] Preserving partial work from truncated attempt")
                else:
                    # Full rollback for genuine logic failures
                    git.restore_head()
                    git.checkpoint(f"retry {tid} attempt {attempt}")
                    tools.fs.revert()  # discard staged changes
                    self.logger.info(f"[{tid}] Full rollback on logic failure")

                # Build lessons learned from previous attempts
                prev_error = retry_errors.get(tid, "")
                if prev_error:
                    lessons_learned = (
                        f"\n\n--- PREVIOUS ATTEMPT ERROR ---\n"
                        f"{prev_error}\n"
                        f"---\n\n"
                        f"Instruction: Do NOT repeat the same mistake. Try a fundamentally "
                        f"different approach. If a test keeps failing with the same error, "
                        f"consider fixing the source code instead of the test."
                    )

            # Inject workspace root and lessons into task context
            task["project_root"] = str(project_root)
            task["workspace_hint"] = (
                f"Project workspace is at: {project_root}\n"
                f"All file paths are relative to this directory.\n"
                f"Do NOT use 'cd /testbed' or similar hardcoded paths.\n"
                f"Run commands directly (e.g. 'python -m pytest tests/')."
            )
            if lessons_learned:
                task["observation"] = task.get("observation", "") + lessons_learned

            task["type"] = "implement"
            impl_result = await self._run_implementer(tools, memory, task)
            # Fix #6B: Log truncation_loop status specifically
            if impl_result["status"] == "truncation_loop":
                self.logger.warning(
                    f"[{tid}] Implementer hit truncation loop. Will preserve partial work on retry."
                )
            elif impl_result["status"] != "complete":
                self.logger.error(f"[{tid}] Implementer failed: {impl_result['result']}")
            retry_errors[tid] = f"Implementer error: {impl_result['result']}"
            if impl_result["status"] != "complete":
                continue

            current_files = impl_result.get("files_changed", [])
            files_changed = list(dict.fromkeys(files_changed + current_files))
            tools.fs.commit(current_files)
            git.checkpoint(f"{tid} implemented")

            # VERIFY
            verify_result = await self._run_verifier(tools, memory, task, current_files, project_root)
            if verify_result["status"] != "complete":
                self.logger.error(f"[{tid}] Verifier did not complete")
                retry_errors[tid] = f"Verifier error: {verify_result.get('result', 'unknown')}"
                continue

            verdict = verify_result.get("verdict", "pass")
            details = verify_result.get("details", {})
            if not isinstance(details, dict):
                details = {}
            issues = details.get("issues", [])

            if verdict == "pass":
                memory.add_task(
                    task_id=tid, description=task["description"],
                    status="complete", files_changed=files_changed,
                    summary=impl_result["result"],
                    duration=time.time() - task_start, attempts=attempt,
                )
                self.logger.info(f"[{tid}] PASS")
                return True

            # Compute error signature for escalation check
            error_sig = ""
            if issues:
                error_sig = " | ".join(str(i) for i in issues[:3])
            elif verify_result.get("result"):
                error_sig = verify_result["result"][:200]

            # ESCALATION: same signature as last attempt -> escalate to repair immediately
            if attempt >= 2 and error_sig and error_sig == prev_error_signature:
                self.logger.info(f"[{tid}] Repeated same error on retry. Escalating to RepairAgent...")
                repair_ok = await self._run_repair(tools, memory, task, issues or [error_sig], current_files, project_root)
                if repair_ok:
                    tools.fs.commit(current_files)
                    git.checkpoint(f"{tid} repaired-escalated")
                    verify2 = await self._run_verifier(tools, memory, task, current_files, project_root)
                    if verify2.get("verdict") == "pass":
                        memory.add_task(
                            task_id=tid, description=task["description"],
                            status="complete", files_changed=files_changed,
                            summary="Repair succeeded (escalated)",
                            duration=time.time() - task_start, attempts=attempt,
                        )
                        self.logger.info(f"[{tid}] PASS after escalated repair")
                        return True
                self.logger.info(f"[{tid}] Escalated repair/re-verify failed, rolling back...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Escalated repair failed. Error: {error_sig}"
                prev_error_signature = error_sig
                continue

            if verdict == "issues" and issues:
                self.logger.info(f"[{tid}] Issues found, dispatching repair...")
                repair_ok = await self._run_repair(tools, memory, task, issues, current_files, project_root)
                if repair_ok:
                    tools.fs.commit(current_files)
                    git.checkpoint(f"{tid} repaired")
                    verify2 = await self._run_verifier(tools, memory, task, current_files, project_root)
                    if verify2.get("verdict") == "pass":
                        memory.add_task(
                            task_id=tid, description=task["description"],
                            status="complete", files_changed=files_changed,
                            summary="Repair succeeded",
                            duration=time.time() - task_start, attempts=attempt,
                        )
                        self.logger.info(f"[{tid}] PASS after repair")
                        return True
                self.logger.info(f"[{tid}] Repair/re-verify failed, rolling back...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Repair failed. Issues: {issues}"
                prev_error_signature = error_sig
                continue

            if verdict == "fail":
                self.logger.info(f"[{tid}] Verdict=fail, rolling back to retry...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Verifier fail: {verify_result.get('result', 'unknown')}"
                prev_error_signature = error_sig
                continue

            self.logger.warning(f"[{tid}] Unknown verdict '{verdict}', retrying...")
            tools.fs.revert(current_files)
            retry_errors[tid] = f"Unknown verdict: {verdict}"

        memory.add_task(
            task_id=tid, description=task["description"],
            status="failed", files_changed=files_changed,
            summary=f"Failed after {max_retries} attempts",
            duration=time.time() - task_start, attempts=attempts,
        )
        return False

    async def _run_implementer(self, tools: ToolRegistry, memory: ProjectMemory, task: dict) -> dict:
        agent = ImplementerAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        return await agent.run(task)

    async def _run_verifier(self, tools: ToolRegistry, memory: ProjectMemory,
                             task: dict, target_files: list[str],
                             project_root: Path) -> dict:
        verify_task = {
            "description": f"Review and test: {task['description']}",
            "target_files": target_files,
            "project_root": str(project_root),
            "type": "verify",
        }
        agent = VerifierAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        return await agent.run(verify_task)

    async def _run_repair(self, tools: ToolRegistry, memory: ProjectMemory,
                           task: dict, issues: list, target_files: list[str],
                           project_root: Path) -> bool:
        repair_task = {
            "description": (
                f"Fix issues in task {task['id']}: {task['description']}\n\n"
                f"Issues: {chr(10).join(f'- {i}' for i in issues)}"
            ),
            "target_files": target_files,
            "project_root": str(project_root),
            "type": "repair",
        }
        agent = RepairAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        result = await agent.run(repair_task)
        return result["status"] == "complete" and result.get("verdict") != "fail"

    async def _setup_venv(self, venv_path: Path):
        if sys.platform == "win32":
            pip = str(venv_path / "Scripts\\pip.exe")
        else:
            pip = str(venv_path / "bin" / "pip")
        loop = asyncio.get_running_loop()
        try:
            if not venv_path.exists():
                self.logger.info(f"Creating venv: {venv_path}")
                await loop.run_in_executor(None, lambda: subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, capture_output=True, text=True,
                ))
                await loop.run_in_executor(None, lambda: subprocess.run(
                    [pip, "install", "--upgrade", "pip"],
                    check=True, capture_output=True, text=True,
                ))
                self.logger.info("venv created")
            
            # Install base dev packages into the venv
            # Note: --break-system-packages is for system Python (PEP 668),
            # NOT for virtual environments. Inside a venv all packages are
            # user-managed by definition, so we never use that flag here.
            await loop.run_in_executor(None, lambda: subprocess.run(
                [pip, "install", "pytest", "pytest-timeout", "ruff", "mypy"],
                check=True, capture_output=True, text=True,
            ))
            
            # Install VPython into the venv
            try:
                await loop.run_in_executor(None, lambda: subprocess.run(
                    [pip, "install", "vpython"],
                    check=True, capture_output=True, text=True,
                ))
                self.logger.info("VPython installed successfully")
            except subprocess.CalledProcessError as exc:
                self.logger.warning(f"VPython installation failed (Python {sys.version_info.major}.{sys.version_info.minor}): {exc}")
                self.logger.info("Installing VPython with compatibility flags...")
                # Try alternative installation methods
                try:
                    await loop.run_in_executor(None, lambda: subprocess.run(
                        [pip, "install", "vpython==7.6.2"],  # Known stable version
                        check=True, capture_output=True, text=True,
                    ))
                    self.logger.info("VPython 7.6.2 installed successfully")
                except subprocess.CalledProcessError as exc2:
                    self.logger.error(f"All VPython installation attempts failed: {exc2}")
                    self.logger.info("VPython will not be available for this project")
                
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"venv setup failed: {exc}")
            raise RuntimeError(f"venv setup failed: {exc}") from exc
        except FileNotFoundError as exc:
            self.logger.error(f"venv pip not found — corrupted or cross-platform issue: {exc}")
            raise RuntimeError(f"venv pip not found: {exc}") from exc
        except Exception as exc:
            self.logger.error(f"venv setup unexpected error: {exc}")
            raise RuntimeError(f"venv setup unexpected error: {exc}") from exc

    def _fail(self, phase: str, error: str, project_root: Path) -> dict:
        self.logger.error(f"Failed in {phase}: {error}")
        return {
            "status": "failed",
            "phase": phase,
            "error": error,
            "project_root": str(project_root),
        }
