"""Structured-output LLM calls through the Anthropic API, or through the local `claude` CLI.

The API is used when ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set. Otherwise the
Claude Code CLI (`claude -p`) is used, which runs on your existing Claude login.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import config


class LLMError(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self, model: str = config.LLM_MODEL, effort: str = "high"):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=4)
        self.model, self.effort = model, effort
        self.label = f"Anthropic API ({model}, effort={effort})"

    def complete_json(self, system: str, prompt: str, schema: dict) -> dict:
        import anthropic

        try:
            # fallbacks="default": if the request is declined by a safety classifier,
            # the API re-runs it on a fallback model inside the same call.
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("request was declined (refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


def _cli_candidates():
    if os.environ.get("IDEAPLOTTER_CLAUDE"):
        yield Path(os.environ["IDEAPLOTTER_CLAUDE"])
    found = shutil.which("claude")
    if found:
        # On Windows npm installs a .cmd/.ps1 shim; call the bundled exe directly so
        # arguments (JSON schema, system prompt) are not re-parsed by cmd.exe.
        exe = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        yield exe if exe.exists() else Path(found)
    # The Claude desktop app keeps its own, usually newer, Claude Code builds.
    home = Path.home()
    for base in (Path(os.environ.get("APPDATA", home)) / "Claude" / "claude-code",
                 home / "Library" / "Application Support" / "Claude" / "claude-code",
                 home / ".config" / "Claude" / "claude-code"):
        yield from base.glob("*/*/claude.exe")
        yield from base.glob("*/*/claude")


def _cli_version(exe: Path) -> tuple:
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30).stdout
        return tuple(int(x) for x in out.split()[0].split("."))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return ()


def _find_claude_cli() -> str | None:
    """The newest Claude Code binary available (new models need recent CLI versions)."""
    versions = {str(p): _cli_version(p) for p in _cli_candidates() if p.exists()}
    versions = {p: v for p, v in versions.items() if v}
    return max(versions, key=versions.get) if versions else None


class ClaudeCLI:
    def __init__(self, model: str = config.LLM_MODEL, effort: str = "high", timeout: int = 900):
        self.exe = _find_claude_cli()
        if not self.exe:
            raise LLMError("Claude Code CLI not found: install it, or set ANTHROPIC_API_KEY to use the API")
        self.model, self.effort, self.timeout = model, effort, timeout
        self.cwd = tempfile.mkdtemp(prefix="ideaplotter-")  # keep project CLAUDE.md etc. out of the context
        self.label = f"Claude Code CLI {'.'.join(map(str, _cli_version(Path(self.exe))))} ({model}, effort={effort})"

    def complete_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            return self._run(system, prompt, schema)
        except LLMError as e:
            if "does not support this model" not in str(e) or self.model == "opus":
                raise
            print(f"  {self.exe} cannot run {self.model}; falling back to the 'opus' alias")
            self.model = "opus"
            self.label = f"Claude Code CLI (opus alias, effort={self.effort})"
            return self._run(system, prompt, schema)

    def _run(self, system: str, prompt: str, schema: dict) -> dict:
        cmd = [
            self.exe, "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--tools", "",
            "--model", self.model,
            "--effort", self.effort,
            "--system-prompt", system,
            "--json-schema", json.dumps(schema, separators=(",", ":")),
        ]
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                               timeout=self.timeout, cwd=self.cwd)
        except subprocess.TimeoutExpired as e:
            raise LLMError("claude CLI timed out") from e
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude CLI returned non-JSON output: {(r.stdout or r.stderr)[:400]}") from e
        if out.get("is_error") or out.get("structured_output") is None:
            raise LLMError(f"claude CLI error: {str(out.get('result') or out)[:400]}")
        return out["structured_output"]


class OpenRouterLLM:
    """Any model on OpenRouter (OpenAI-compatible chat API) with JSON-schema structured output."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str = config.OPENROUTER_MODEL, timeout: int = 180):
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise LLMError("OPENROUTER_API_KEY is not set (get one at https://openrouter.ai/keys)")
        self.model, self.timeout = model, timeout
        self.label = f"OpenRouter ({model})"
        self.cost, self.calls = 0.0, 0  # running totals, reported by OpenRouter

    def complete_json(self, system: str, prompt: str, schema: dict) -> dict:
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "game_idea", "strict": True, "schema": schema}},
            "provider": {"require_parameters": True},  # only route to providers that honour the schema
            "max_tokens": 8000,  # reasoning models (e.g. gpt-oss) spend tokens thinking before the JSON
            "usage": {"include": True},
        }
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ideaplotter", "X-Title": "IdeaPlotter"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504):
                    last = f"HTTP {e.code}: {detail}"
                    time.sleep(2 * 2 ** attempt)
                    continue
                raise LLMError(f"OpenRouter HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = str(e)
                time.sleep(2 * 2 ** attempt)
        else:
            raise LLMError(f"OpenRouter failed after retries: {last}")
        if out.get("error"):
            raise LLMError(f"OpenRouter error: {str(out['error'])[:300]}")
        self.cost += float((out.get("usage") or {}).get("cost") or 0)
        self.calls += 1
        choice = out["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError("response hit max_tokens")
        text = (choice["message"].get("content") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        return json.loads(text)


def make(backend: str = "auto", model: str = config.LLM_MODEL, effort: str = "high"):
    """api = Anthropic SDK, cli = local Claude Code, openrouter = any OpenRouter model.

    auto picks the Anthropic API if a key is set, then OpenRouter if OPENROUTER_API_KEY is set,
    then the Claude Code CLI."""
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if backend == "api" or (backend == "auto" and has_key):
        return AnthropicLLM(model, effort)
    if backend == "openrouter" or (backend == "auto" and os.environ.get("OPENROUTER_API_KEY")):
        return OpenRouterLLM(config.OPENROUTER_MODEL if model == config.LLM_MODEL else model)
    if backend in ("cli", "auto"):
        return ClaudeCLI(model, effort)
    raise ValueError(f"unknown backend {backend!r}")
