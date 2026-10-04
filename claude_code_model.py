"""An Inspect model provider that runs Claude through the `claude` CLI, so evals
use your Claude subscription instead of API credits.

Usage:
    uv run inspect eval task.py --model claudecode/sonnet
    uv run inspect eval task.py --model claudecode/opus -M effort=high

How it works: each turn, Inspect calls generate() with the whole conversation.
We render it as one text prompt (plus the tool list), run `claude -p` once, and
force the answer into a JSON schema {message, tool, arguments}. That answer
becomes an ordinary Inspect tool call, so the react() loop, tools, turn limit,
logs and replay.py all work unchanged.

Like bot.py, the model is stateless: every call re-sends the full history.

Caveat: this is not native tool calling. The model writes its tool call as
structured JSON, which is close to, but not the same as, Claude via the API.
"""

import asyncio
import json
import os
import tempfile
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolFunction, ToolInfo

NO_TOOL = "none"  # the "tool" value for a plain reply with no tool call
TIMEOUT = 300  # seconds per CLI call


class ClaudeCLIError(RuntimeError):
    def __init__(self, message: str, retry: bool) -> None:
        super().__init__(message)
        self.retry = retry


class ClaudeCodeAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        effort: str | None = None,  # -M effort=low|medium|high
        cli: str = "claude",  # -M cli=/path/to/claude
        # -M harness=json: play the game one `claude -p` call per turn, with the
        # tools described in the prompt (the first version of this harness).
        # The default, "mcp", plays it as one session with native MCP tools.
        harness: str = "mcp",
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        self.effort = effort
        self.harness = harness
        self.cli = cli

    def max_connections(self) -> int:
        # Each connection is a CLI process on your subscription; keep it modest.
        return 4

    def should_retry(self, ex: Exception) -> bool:
        # Inspect retries the call with backoff when this returns True.
        return isinstance(ex, ClaudeCLIError) and ex.retry

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        system = "\n\n".join(m.text for m in input if isinstance(m, ChatMessageSystem))
        choices = tool_names(tools, tool_choice)
        prompt = render_prompt(input, tools, choices)
        schema = answer_schema(choices)

        args = [
            self.cli, "-p",
            "--model", self.model_name,
            "--system-prompt", system or "You are a helpful assistant.",
            "--json-schema", json.dumps(schema),
            "--output-format", "json",
            # No Claude Code tools, settings, MCP servers, skills or saved
            # session: the model sees only our system prompt and our prompt.
            "--tools", "",
            "--no-session-persistence",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
        ]
        if self.effort:
            args += ["--effort", self.effort]

        # Without the API key in the environment, the CLI uses the logged-in
        # subscription rather than billing the API.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tempfile.gettempdir(),  # away from any CLAUDE.md in the project
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(prompt.encode()), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            raise ClaudeCLIError(f"claude timed out after {TIMEOUT}s", retry=True)

        details = (
            f"exit {process.returncode}, stderr: {err.decode()[:500]!r}, "
            f"stdout: {out.decode()[:500]!r}"
        )
        try:
            result = json.loads(out.decode())
        except json.JSONDecodeError:
            # Empty or garbled output: the CLI itself failed. Seen occasionally
            # under load; a retry usually succeeds.
            raise ClaudeCLIError(f"claude gave no JSON ({details})", retry=True)
        if not isinstance(result, dict) or "subtype" not in result:
            raise ClaudeCLIError(f"claude gave unexpected JSON ({details})", retry=True)

        call = ModelCall.create(
            request={"args": args, "prompt": prompt},
            response=result,
        )
        answer = result.get("structured_output")
        if result.get("is_error") or not isinstance(answer, dict):
            # Not retried: a refusal by Anthropic's safeguards (the same
            # conversation would hit the same check), or a subscription usage
            # limit (it lasts hours; rerun failed games with eval-retry later).
            text = str(result.get("result"))
            final = any(s in text.lower() for s in ("safeguards", "hit your", "usage limit"))
            raise ClaudeCLIError(
                f"claude: {result.get('subtype')}: {text[:500]}",
                retry=not final,
            )

        message = str(answer.get("message", ""))
        tool = answer.get("tool", NO_TOOL)
        if tool == NO_TOOL:
            output = ModelOutput.from_content(self.model_name, message)
        else:
            output = ModelOutput.for_tool_call(
                self.model_name, tool, answer.get("arguments") or {}, content=message
            )
        output.usage = usage(result.get("usage") or {})
        return output, call


@modelapi(name="claudecode")
def claudecode() -> type[ModelAPI]:
    return ClaudeCodeAPI


# --- Prompt and schema -------------------------------------------------------


def tool_names(tools: list[ToolInfo], tool_choice: ToolChoice) -> list[str]:
    """The values the model may put in "tool"."""
    if isinstance(tool_choice, ToolFunction):
        return [tool_choice.name]
    if tool_choice == "none" or not tools:
        return [NO_TOOL]
    names = [t.name for t in tools]
    return names if tool_choice == "any" else names + [NO_TOOL]


def answer_schema(choices: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "What you say to the user this turn (shown alongside any tool call).",
            },
            "tool": {"type": "string", "enum": choices},
            "arguments": {
                "type": "object",
                "description": "Arguments for the tool, matching its parameters. {} if none.",
            },
        },
        "required": ["message", "tool", "arguments"],
    }


def render_prompt(input: list[ChatMessage], tools: list[ToolInfo], choices: list[str]) -> str:
    parts = []

    if tools and choices != [NO_TOOL]:
        lines = ["# Tools you can call"]
        for t in tools:
            params = t.parameters.model_dump(exclude_none=True)
            lines.append(f"- {t.name}: {t.description}\n  parameters: {json.dumps(params)}")
        parts.append("\n".join(lines))

    lines = ["# Conversation so far"]
    for m in input:
        if isinstance(m, ChatMessageUser):
            lines.append(f"[user]\n{m.text}")
        elif isinstance(m, ChatMessageAssistant):
            text = f"[you]\n{m.text}" if m.text else "[you]"
            for c in m.tool_calls or []:
                text += f"\n-> {c.function}({json.dumps(c.arguments)})"
            lines.append(text)
        elif isinstance(m, ChatMessageTool):
            result = m.error.message if m.error else m.text
            lines.append(f"[result of {m.function}]\n{result}")
    parts.append("\n\n".join(lines))

    if choices == [NO_TOOL]:
        parts.append('Reply now. Put your reply in "message" and set "tool" to "none".')
    elif NO_TOOL in choices:
        parts.append(
            "Your turn. Call exactly one tool by filling in \"tool\" and \"arguments\", "
            'or set "tool" to "none" to reply without calling a tool.'
        )
    else:
        parts.append('Your turn. Call exactly one tool by filling in "tool" and "arguments".')
    return "\n\n".join(parts)


def usage(raw: dict) -> ModelUsage:
    input_tokens = int(raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_cache_read=cache_read,
        input_tokens_cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
    )
