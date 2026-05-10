"""OpenAI-compatible LLM client with tool calling support + async + caching."""

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx


class LLMError(Exception):
    pass


def _hash_request(messages: list[dict], tools: list[dict] | None, temperature: float,
                  model: str = "", max_tokens: int = 0, thinking_budget: int = 0) -> str:
    payload = {"m": messages, "t": tools, "temp": temperature,
               "model": model, "max_tok": max_tokens, "think": thinking_budget}
    data = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:24]


class LLMBackend:
    def __init__(self, base_url: str, model: str, api_key: str, context_size: int,
                 reasoning_format: str, temperature: float, max_tokens: int, logger,
                 cache_dir: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.context_size = context_size
        self.reasoning_format = reasoning_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        headers = {}
        if api_key and api_key != "no-key-needed":
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(7200.0, connect=30.0),
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self._memory_cache: dict[str, dict] = {}
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        # Circuit breaker state (Fix #5C)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    async def achat(self, messages: list[dict], tools: list[dict] | None = None,
                    agent_name: str = "unknown", thinking_budget: int = 2048,
                    stream: bool = True, on_token=None) -> dict:
        cache_key = _hash_request(messages, tools, self.temperature,
                                   model=self.model, max_tokens=self.max_tokens,
                                   thinking_budget=thinking_budget)
        cached = self._memory_cache.get(cache_key)
        if cached:
            self.logger.debug(f"LLM cache hit for {agent_name}")
            return cached
        if self._cache_dir:
            cache_file = self._cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                self._memory_cache[cache_key] = cached
                self.logger.debug(f"LLM disk cache hit for {agent_name}")
                return cached

        # Default token callback: print live to stdout (respect logger settings)
        if on_token is None:
            def _default_on_token(token: str):
                # Live raw token printing disabled — structured output is emitted
                # by logger.llm_call() after the stream completes.
                return
            on_token = _default_on_token

        # Circuit breaker check (Fix #5C)
        if self._consecutive_failures >= 5 and time.time() < self._circuit_open_until:
            raise LLMError("Circuit breaker open — LLM API appears down")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if self.reasoning_format == "deepseek":
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        start = time.time()
        error = None
        data = None
        for attempt in range(3):
            try:
                if stream:
                    data = await self._achat_stream(
                        payload, agent_name=agent_name, on_token=on_token
                    )
                else:
                    resp = await self.client.post(
                        f"{self.base_url}/chat/completions", json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ConnectTimeout,
                    httpx.ReadTimeout, httpx.NetworkError) as exc:
                # Log the server error body on HTTPStatusError so we can see WHY it failed
                error_detail = ""
                if isinstance(exc, httpx.HTTPStatusError) and exc.response:
                    try:
                        body = exc.response.text[:800]
                        error_detail = f" | body={body}"
                    except Exception as exc:
                        self.logger.debug(f"Failed to read error body: {exc}")
                # Fix #5A: Add jitter to backoff delays
                base = 2 ** (attempt + 1)
                jitter = random.uniform(0.5, 1.5)  # ±50% jitter
                wait = base * jitter
                if attempt < 2:
                    self.logger.warning(
                        f"LLM failed ({exc.__class__.__name__}{error_detail}), attempt {attempt+2}/3 in {wait:.1f}s",
                        agent=agent_name,
                    )
                    await asyncio.sleep(wait)
                else:
                    error = exc
        else:
            # All 3 retries failed (Fix #5C: update circuit breaker)
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                self._circuit_open_until = time.time() + 60  # 60s cooldown
            # Include last error body in the final exception if available
            final_detail = ""
            if isinstance(error, httpx.HTTPStatusError) and error.response:
                try:
                    final_detail = f" | last body: {error.response.text[:800]}"
                except Exception as exc:
                    self.logger.debug(f"Failed to read final error body: {exc}")
            raise LLMError(f"LLM at {self.base_url} unreachable after 3 attempts: {error}{final_detail}")

        duration = time.time() - start
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        thinking = choice.get("reasoning_content") or ""
        tool_calls_raw = choice.get("tool_calls") or []
        usage = data.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)

        # Fix #1A: Extract finish_reason from API response
        finish_reason = data["choices"][0].get("finish_reason", "stop")

        normalized_tool_calls = self._normalize_tool_calls(tool_calls_raw)

        self.logger.llm_call(
            agent=agent_name, messages=messages,
            response=content, thinking=thinking,
            tokens_prompt=prompt_tok, tokens_completion=comp_tok,
            duration=duration,
            finish_reason=finish_reason,
            streamed=stream,
            tool_calls=normalized_tool_calls,
        )

        # Reset circuit breaker on success (Fix #5C)
        self._consecutive_failures = 0

        result = {
            "content": content,
            "thinking": thinking,
            "tool_calls": self._normalize_tool_calls(tool_calls_raw),
            "tokens_prompt": prompt_tok,
            "tokens_completion": comp_tok,
            "finish_reason": finish_reason,  # Fix #1A
            "truncated": finish_reason == "length",  # Fix #1A
        }

        self._memory_cache[cache_key] = result
        if self._cache_dir:
            (self._cache_dir / f"{cache_key}.json").write_text(
                json.dumps(result, default=str), encoding="utf-8"
            )
        return result

    def _normalize_tool_calls(self, raw: list[dict]) -> list[dict]:
        import uuid
        out = []
        for tc in raw:
            fn = tc.get("function", {})
            name = (fn.get("name") or "").strip()
            args_raw = fn.get("arguments", "{}")
            malformed = False

            # Handle None, empty string, or whitespace-only arguments.
            # Truncated responses often produce tool calls with empty args.
            if args_raw is None or (isinstance(args_raw, str) and not args_raw.strip()):
                self.logger.warning(
                    f"Empty tool call arguments for '{name}' — "
                    f"this usually indicates a truncated or malformed response."
                )
                args = {}
                malformed = True
            elif isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    # Fix #1B: Log warning instead of silently falling back
                    self.logger.warning(
                        f"Malformed tool call arguments for '{name}': {e}. "
                        f"Raw preview: {args_raw[:200]}"
                    )
                    args = {}
                    malformed = True
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            # Fix #2B: Use globally unique IDs (server-provided or UUID) to
            # prevent collisions across different LLM calls. Duplicate IDs like
            # "tc_0" cause _validate_message_flow to incorrectly match orphaned
            # tool-role messages to wrong assistant messages.
            tc_id = tc.get("id")
            if not tc_id or tc_id.startswith("tc_"):
                tc_id = f"tc_{uuid.uuid4().hex[:12]}"
            out.append({
                "id": tc_id,
                "name": name,
                "arguments": args,
                "_malformed": malformed,  # Fix #1B: Flag for agent loop
            })
        return out

    async def _achat_stream(self, payload: dict, agent_name: str, on_token=None) -> dict:
        """Stream from the LLM via SSE and accumulate into a standard response dict."""
        import json as _json

        content = ""
        thinking = ""
        tool_calls: list[dict] = []
        finish_reason: str | None = None
        usage: dict = {}

        async with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason") or finish_reason

                # Accumulate content and fire callback
                delta_content = delta.get("content")
                if delta_content:
                    content += delta_content
                    if on_token:
                        on_token(delta_content)

                # Accumulate reasoning tokens (DeepSeek-style)
                delta_thinking = delta.get("reasoning_content") or delta.get("thinking")
                if delta_thinking:
                    thinking += delta_thinking

                # Accumulate tool calls incrementally
                delta_tcs = delta.get("tool_calls")
                if delta_tcs:
                    for tc_delta in delta_tcs:
                        idx = tc_delta.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        tc = tool_calls[idx]
                        if tc_delta.get("id"):
                            tc["id"] = tc_delta["id"]
                        if tc_delta.get("type"):
                            tc["type"] = tc_delta["type"]
                        fn_delta = tc_delta.get("function", {})
                        if fn_delta.get("name"):
                            tc["function"]["name"] = fn_delta["name"]
                        if fn_delta.get("arguments"):
                            tc["function"]["arguments"] += fn_delta["arguments"]

                # Some APIs emit usage only in the final chunk
                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    usage = chunk_usage

        message: dict = {"role": "assistant", "content": content}
        if thinking:
            message["reasoning_content"] = thinking
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "choices": [
                {"message": message, "finish_reason": finish_reason or "stop"}
            ],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }

    async def close(self):
        await self.client.aclose()