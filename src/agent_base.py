"""Base agent - ReAct loop with native tool calling only."""

import asyncio
import json
import re
from pathlib import Path


class AgentBase:
    """Think -> Tool call(s) -> Observe -> Repeat -> task_complete."""

    agent_role: str = "base"
    system_prompt_path: str = "base.txt"

    # Maximum messages to keep in history (system + recent context).
    # When exceeded, older messages are dropped (but system prompt is always kept).
    # This is a HARD upper bound — token-based trimming (see _trim_messages) will
    # activate well before this limit in normal operation.
    MAX_MESSAGES = 50

    # Fraction of the context window to reserve for the LLM's response.
    # When estimated prompt tokens exceed (1 - RESERVE_FRACTION) * context_size,
    # older messages are trimmed. This prevents context overflow deaths.
    RESERVE_FRACTION = 0.20  # reserve 20% for response + overhead

    # Rough token estimation: ~4 chars per token for English/code.
    # This is intentionally conservative (underestimates tokens) so we trim
    # earlier rather than later.
    CHARS_PER_TOKEN = 3.5

    # Number of consecutive iterations with the same error pattern before
    # forcing a task_complete with verdict='issues'.
    CONVERGENCE_THRESHOLD = 3

    # Safety margin: subtract this many tokens from the prompt budget to
    # account for estimation noise, message overhead (role tags, tool call
    # wrappers), and model-specific tokenization differences. Prevents edge
    # cases where the heuristic estimate says "safe" but the server rejects.
    SAFETY_MARGIN = 4096

    def __init__(self, llm, tools, memory, logger, prompts_dir: str,
                 max_iterations: int = 15, thinking_budget: int = 2048):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.max_iterations = max_iterations
        self.thinking_budget = thinking_budget
        self._consecutive_truncations = 0  # Fix #1E: Track truncation streaks
        self._token_estimation_multiplier = 1.0  # Calibrated against real usage

        prompt_file = Path(prompts_dir) / self.system_prompt_path
        if prompt_file.exists():
            self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()
        else:
            self.system_prompt = f"You are the {self.agent_role} agent."
            logger.warning(f"Prompt file missing: {prompt_file}")

    def build_system_message(self) -> str:
        parts = [self.system_prompt]
        parts.append("\n\n--- PROJECT MEMORY ---")
        parts.append(self.memory.get_compact())
        parts.append("---")
        parts.append("\n\n--- AVAILABLE TOOLS ---")
        for t in self.tools.get_schemas(self.agent_role):
            fn = t["function"]
            params = ", ".join(fn["parameters"]["properties"].keys())
            parts.append(f"  {fn['name']}({params}) — {fn['description']}")
        parts.append("---")
        return "\n".join(parts)

    def build_initial_messages(self, task: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.build_system_message()},
            {"role": "user", "content": self._format_task(task)},
        ]

    def _format_task(self, task: dict) -> str:
        project = self.memory.data.get("project", {}).get("name", "this project")
        lines = [
            f"You are working on project: {project}",
            "",
            f"Task: {task.get('description', '')}",
        ]
        if task.get("type"):
            lines.append(f"Type: {task['type']}")
        if task.get("workspace_hint"):
            lines.append(f"\n--- WORKSPACE ---\n{task['workspace_hint']}\n---")
        if task.get("observation"):
            lines.append(f"\n--- OBSERVED STATE ---\n{task['observation']}\n---")
        if task.get("target_files"):
            lines.append(f"Target files: {', '.join(task['target_files'])}")
        if task.get("acceptance"):
            lines.append(f"Acceptance: {task['acceptance']}")
        return "\n".join(lines)

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for a message list.

        Uses a conservative chars-per-token ratio. Tool call arguments and
        tool results tend to be verbose (file contents, command output), so
        we sum all text fields in each message.
        """
        total_chars = 0
        for msg in messages:
            # Content field
            content = msg.get("content", "")
            if content:
                total_chars += len(content)
            # Tool calls (arguments JSON)
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    total_chars += len(args)
                elif isinstance(args, dict):
                    total_chars += len(json.dumps(args))
                total_chars += len(fn.get("name", ""))
            # Tool result
            if msg.get("role") == "tool":
                total_chars += len(msg.get("content", ""))
        # Conservative estimate: divide by CHARS_PER_TOKEN, adjusted by
        # runtime calibration from actual API usage.
        return max(1, int(total_chars / self.CHARS_PER_TOKEN * self._token_estimation_multiplier))

    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        """Trim message history to stay within the context window budget.

        Strategy:
        1. Always keep the system message (index 0).
        2. Always keep the first user message (index 1) — the LLM needs a
           user query to generate a response.
        3. Never drop all user messages from the remaining history.
        4. Keep as many recent messages as fit within the token budget.
        5. Never exceed MAX_MESSAGES as a hard upper bound.
        6. Validate message flow: ensure every tool-role message has a preceding
           assistant message with a matching tool_call_id.
        """
        if not messages:
            return messages

        system_msg = messages[0]

        # Determine the token budget for prompts (reserve fraction for response)
        context_size = getattr(self.llm, "context_size", 32768)
        max_prompt_tokens = int(context_size * (1 - self.RESERVE_FRACTION) - self.SAFETY_MARGIN)

        # Fast path: if total messages are few and short, skip token estimation
        if len(messages) <= 5:
            estimated = self._estimate_tokens(messages)
            if estimated <= max_prompt_tokens:
                # Validate message flow even in fast path
                return self._validate_message_flow(messages)

        # Estimate tokens for the full history
        full_estimated = self._estimate_tokens(messages)

        if full_estimated <= max_prompt_tokens and len(messages) <= self.MAX_MESSAGES + 1:
            # Everything fits — no trimming needed, but validate flow
            return self._validate_message_flow(messages)

        # --- Trimming strategy: drop from the MIDDLE, keep head + tail ---
        # The first user message (index 1) is sacred — it contains the task.
        # The most recent messages are sacred — they contain current state.
        # We drop oldest messages *after* index 1, working backwards from
        # the middle, until we fit within the budget.

        system_tokens = self._estimate_tokens([system_msg])
        head = messages[0:2]   # system + first user
        tail = messages[2:]   # everything after first user

        # Try keeping progressively more of the tail (from the end)
        for keep_count in range(min(len(tail), self.MAX_MESSAGES - 1), -1, -1):
            candidate = head + tail[-keep_count:] if keep_count > 0 else head
            # Ensure at least one user message remains
            has_user = any(m.get("role") == "user" for m in candidate)
            if not has_user:
                # Should never happen because head[1] is user, but be defensive
                continue
            estimated = self._estimate_tokens(candidate)
            if estimated <= max_prompt_tokens:
                dropped = len(messages) - len(candidate)
                if dropped > 0:
                    self.logger.info(
                        f"[{self.agent_role}] Message history trimmed from "
                        f"{len(messages)} to {len(candidate)} "
                        f"(dropped {dropped} middle messages, "
                        f"estimated {estimated} tokens <= {max_prompt_tokens} budget)"
                    )
                # Validate message flow after trimming
                return self._validate_message_flow(candidate)

        # Extreme case: even system + first user exceeds budget.
        # Keep system + first user as absolute minimum.
        self.logger.warning(
            f"[{self.agent_role}] Context severely over budget "
            f"(system+user={self._estimate_tokens(head)} estimated tokens, "
            f"budget={max_prompt_tokens}). Keeping system + first user only."
        )
        return self._validate_message_flow(head)

    def _validate_message_flow(self, messages: list[dict]) -> list[dict]:
        """Validate message flow: ensure every tool-role message has a preceding
        assistant message with a matching tool_call_id. Drop tool-role messages
        without a matching assistant tool call to prevent history corruption.
        
        Fix: enforce STRICT ORDERING — tool-role messages must appear AFTER
        their matching assistant. This prevents collision bugs where trimmed
        messages produce tool-before-assistant sequences that pass a naive
        set-membership check but fail the LLM server's sequential validation.
        """
        valid_messages = []
        # Map tool_call_id -> index of the assistant message that owns it.
        # Tool-role messages are only valid if they appear AFTER their owner.
        tool_call_positions: dict[str, int] = {}

        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_call_positions[tc["id"]] = len(valid_messages)
                valid_messages.append(msg)
            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                owner_pos = tool_call_positions.get(tc_id)
                # Must be owned AND owner must precede this tool message
                if owner_pos is not None and owner_pos < len(valid_messages):
                    valid_messages.append(msg)
                else:
                    self.logger.warning(
                        f"[{self.agent_role}] Dropped tool-role message with "
                        f"unmatched or out-of-order tool_call_id: "
                        f"{tc_id} (owner at index {owner_pos}, "
                        f"tool at {len(valid_messages)})"
                    )
            else:
                valid_messages.append(msg)

        return valid_messages

    def _extract_error_pattern(self, tool_results: list[dict]) -> str:
        """Extract a short error signature from tool results for convergence detection.

        Returns a normalized string that captures the essence of any errors,
        or empty string if no errors found. Used to detect when the agent is
        stuck in a loop making the same mistake.
        
        Fix: Differentiate between environment errors (path issues) and logic errors.
        """
        patterns = []
        for tr in tool_results:
            result = tr.get("result", "")
            if not result:
                continue
            
            # Check for environment-specific errors that should not trigger convergence
            if "No such file or directory" in result:
                # This is likely an environment/path issue, not a repeated coding mistake
                continue
            if "cannot access" in result and "No such file or directory" in result:
                # Path-related errors
                continue
            if "bin//home" in result:
                # Double path issue - environment problem
                continue
                
            # Look for common error indicators
            if "Exit code:" in result:
                # Extract exit code line — only treat non-zero as errors
                for line in result.split("\n"):
                    if line.startswith("Exit code:"):
                        code_str = line.split(":", 1)[1].strip()
                        try:
                            code = int(code_str)
                        except ValueError:
                            code = -1
                        if code != 0:
                            patterns.append(line.strip())
                        break
            if "Error" in result or "error" in result or "FAILED" in result:
                # Extract first error line, truncate to avoid matching on variable parts
                for line in result.split("\n"):
                    line = line.strip()
                    if "Error" in line or "FAILED" in line:
                        # Normalize: keep first 120 chars to avoid variable line numbers
                        patterns.append(line[:120])
                        break
        return " | ".join(patterns) if patterns else ""

    async def run(self, task: dict) -> dict:
        """Main ReAct loop. Returns dict with status, result, verdict, etc."""
        messages = self.build_initial_messages(task)
        files_changed: set[str] = set()
        empty_retries = 0

        # Track useful iterations separately from total LLM calls
        iteration = 0
        useful_iterations = 0

        # Convergence detection: track recent error patterns
        recent_error_patterns: list[str] = []

        # Fix #1E: Reset truncation counter at start of run
        self._consecutive_truncations = 0

        self.logger.info(f"[{self.agent_role}] Starting task: {task.get('description', '')[:60]}...")

        while useful_iterations < self.max_iterations:
            iteration += 1

            self.logger.info(f"Iteration: {iteration}")

            # Trim message history if it exceeds the budget
            messages = self._trim_messages(messages)

            try:
                response = await self.llm.achat(
                    messages=messages,
                    tools=self.tools.get_schemas(self.agent_role),
                    agent_name=self.agent_role,
                    thinking_budget=self.thinking_budget,
                )
            except Exception as exc:
                self.logger.error(f"[{self.agent_role}] LLM call failed: {exc}")
                return self._result("error", f"LLM failure: {exc}", files_changed, iteration)

            content = response["content"]
            tool_calls = response.get("tool_calls", [])

            # --- Calibration: adjust token estimator against real usage ---
            actual_prompt_tok = response.get("tokens_prompt", 0)
            if actual_prompt_tok > 0 and messages:
                estimated = self._estimate_tokens(messages)
                # Only calibrate on non-trivial prompts to avoid div-by-zero noise
                if estimated > 1000:
                    actual_ratio = actual_prompt_tok / estimated
                    # Exponential moving average: alpha=0.3 blends in new observation
                    alpha = 0.3
                    old_mult = self._token_estimation_multiplier
                    self._token_estimation_multiplier = (
                        old_mult * (1 - alpha) + actual_ratio * alpha
                    )
                    self.logger.debug(
                        f"[{self.agent_role}] Token calibration: "
                        f"estimated={estimated}, actual={actual_prompt_tok}, "
                        f"ratio={actual_ratio:.3f}, multiplier={self._token_estimation_multiplier:.3f}"
                    )

            # --- Truncation detection ---
            truncated = response.get("truncated", False)
            finish_reason = response.get("finish_reason", "stop")

            if truncated and tool_calls:
                # Response was truncated mid-generation and contains incomplete
                # tool calls. Do NOT execute them — inject a message instead.
                self._consecutive_truncations += 1
                self.logger.warning(
                    f"[{self.agent_role}] Response truncated (finish_reason=length). "
                    f"Skipping {len(tool_calls)} tool call(s). "
                    f"Consecutive truncations: {self._consecutive_truncations}"
                )

                # If we've hit 3+ consecutive truncations, abort early
                if self._consecutive_truncations >= 3:
                    return self._result(
                        "truncation_loop",
                        f"Aborted after {self._consecutive_truncations} consecutive truncated responses. "
                        f"Increase max_tokens or reduce thinking_budget in config.",
                        files_changed, iteration,
                    )

                # Tell the LLM to try again with shorter output
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was truncated because it exceeded the "
                        "maximum token limit. Please try again with a SHORTER response. "
                        "If writing a file, break it into smaller pieces or use edit_file "
                        "instead of write_file for large content."
                    ),
                })
                continue

            # Check for malformed tool calls (JSON parse failed but not truncated)
            has_malformed = any(tc.get("_malformed") for tc in tool_calls)
            if has_malformed:
                self._consecutive_truncations += 1
                self.logger.warning(
                    f"[{self.agent_role}] Malformed tool call arguments detected. "
                    f"Skipping execution. Consecutive: {self._consecutive_truncations}"
                )
                if self._consecutive_truncations >= 3:
                    return self._result(
                        "truncation_loop",
                        "Aborted after 3 consecutive malformed tool calls.",
                        files_changed, iteration,
                    )
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous tool call had malformed arguments (JSON parse error). "
                        "This usually happens when the response is too long. Please try "
                        "again with a SHORTER response. Break large files into smaller pieces."
                    ),
                })
                continue

            # Reset truncation counter on successful response
            self._consecutive_truncations = 0

            if not tool_calls and not content.strip():
                empty_retries += 1
                if empty_retries >= 3:
                    return self._result("error", "Repeated empty responses", files_changed, iteration)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Your response was empty. Please use a tool or provide content."})
                continue
            empty_retries = 0

            # If no tool calls, it's a final answer
            if not tool_calls:
                return self._result("complete", content, files_changed, iteration)

            # Log assistant with tool calls
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in tool_calls
                ]
            messages.append(msg)

            # Execute all tool calls and build tool responses
            completion_args = None
            tool_results = []

            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                result = self.tools.execute(name, args, agent_name=self.agent_role)
                tool_results.append({"id": tc["id"], "name": name, "result": result})

                if name in ("write_file", "edit_file") and args.get("path"):
                    files_changed.add(args["path"])

                if name == "task_complete":
                    completion_args = args

            # Add tool results as tool-role messages
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["id"],
                    "content": tr["result"],
                })

            if completion_args:
                verdict = completion_args.get("verdict", "complete")
                summary = completion_args.get("summary", content)
                details = completion_args.get("details", {})
                return {
                    "status": "complete",
                    "result": summary,
                    "verdict": verdict,
                    "details": details,
                    "files_changed": sorted(files_changed),
                    "iterations": iteration,
                }

            # --- Convergence detection ---
            # Track error patterns from tool results. If the same pattern
            # appears CONVERGENCE_THRESHOLD times in a row, force the agent
            # to self-report being stuck via task_complete.
            error_pattern = self._extract_error_pattern(tool_results)
            recent_error_patterns.append(error_pattern)
            if len(recent_error_patterns) > self.CONVERGENCE_THRESHOLD:
                recent_error_patterns.pop(0)

            # Check if we're stuck on the same error
            if (len(recent_error_patterns) >= self.CONVERGENCE_THRESHOLD
                    and all(recent_error_patterns)
                    and len(set(recent_error_patterns)) == 1):
                self.logger.warning(
                    f"[{self.agent_role}] Convergence detected: same error pattern "
                    f"for {self.CONVERGENCE_THRESHOLD} consecutive iterations. "
                    f"Aborting agent loop."
                )
                return self._result(
                    "convergence_loop",
                    f"Agent got stuck repeating the same error pattern for "
                    f"{self.CONVERGENCE_THRESHOLD} consecutive iterations. "
                    f"Last error signature: {recent_error_patterns[-1]}",
                    files_changed, iteration,
                )

            # Fix #2: Only increment useful_iterations for productive iterations
            useful_iterations += 1

        # Report useful vs total iteration counts
        return self._result("max_iterations",
            f"Hit {self.max_iterations} useful iterations ({iteration} total calls)",
            files_changed, iteration)

    def _result(self, status: str, result: str, files_changed: set, iterations: int) -> dict:
        # Fix #6B: For truncation_loop status, use a clear verdict
        verdict = "truncation_loop" if status == "truncation_loop" else status
        return {
            "status": status,
            "result": result,
            "verdict": verdict,
            "details": {},
            "files_changed": sorted(files_changed),
            "iterations": iterations,
        }