"""Play a whole game as one Claude Code session, with the game tools as real tools.

Why: claude_code_model.py runs one `claude -p` call per turn and describes the
game tools in prose, asking for the chosen call as JSON. The model's only real
tool is then Claude Code's structured-output tool, and late in long games it
sometimes concludes that look/move/dig "aren't available" and gives up.

Here the game tools are served by a small MCP server, one per sample, running
inside the Inspect process. Claude Code connects to it, so look, move, dig and
scan_world are genuine tools: the model calls them natively, sees real tool
results, and keeps the whole game in one session.

Inspect's react() loop is not used on this path. We run `claude -p` once per
game, read its stream of messages, and convert them back into Inspect messages
so the scorer, the log and replay.py work as before.

Differences from the API path worth knowing when comparing results:
- Claude Code names MCP tools "mcp__game__<tool>"; we map them back.
- There is no submit() tool. The game ends when the model replies without a
  tool call (or runs out of turns); that last reply is its summary.
- Each tool call is one turn. Calls past the budget are refused. Every tool
  result ends with the turns left (world.spend_turn), as on the API path.
"""

import asyncio
import json
import os
import socket
import tempfile
from contextlib import nullcontext
from typing import Annotated, Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolDef

from inspect_ai.util import store_as

from world import Direction, WorldState, dig, look, move, scan_world

SERVER = "game"  # Claude Code calls our tools mcp__game__<name>
PREFIX = f"mcp__{SERVER}__"
GAME_TIMEOUT = 1800  # seconds for a whole game

# Replaces task.py's ASSISTANT_PROMPT, which refers to react()'s submit() tool.
ENDING = """\
Think briefly about your plan before each action. When you have finished,
reply without calling a tool, with a short summary of what you did."""


class _QuietServer(uvicorn.Server):
    # uvicorn normally takes over Ctrl-C; many of these run inside Inspect.
    def capture_signals(self):
        return nullcontext()


def game_server() -> MCPServer:
    """An MCP server exposing the game tools.

    The handlers call the Inspect tools from world.py directly. They run in
    tasks started from the sample's solver, so they see that sample's Store.
    """
    server = MCPServer(SERVER)
    tools = {t.name: t for t in (ToolDef(f()) for f in (look, move, dig, scan_world))}
    direction_help = tools["move"].parameters.properties["direction"].description

    async def run(name: str, **arguments: Any) -> str:
        # The tools count turns themselves (world.spend_turn); we only stop
        # calls once the budget set by init_world() is spent.
        world = store_as(WorldState)
        if world.max_turns is not None and world.turns_used >= world.max_turns:
            return "You have no turns left. The game is over."
        return str(await tools[name].tool(**arguments))

    @server.tool(name="look", description=tools["look"].description, structured_output=False)
    async def look_tool() -> str:
        return await run("look")

    @server.tool(name="move", description=tools["move"].description, structured_output=False)
    async def move_tool(
        direction: Annotated[Direction, Field(description=direction_help)],
    ) -> str:
        return await run("move", direction=direction)

    @server.tool(name="dig", description=tools["dig"].description, structured_output=False)
    async def dig_tool(
        direction: Annotated[Direction, Field(description=direction_help)],
    ) -> str:
        return await run("dig", direction=direction)

    @server.tool(name="scan_world", description=tools["scan_world"].description, structured_output=False)
    async def scan_tool() -> str:
        return await run("scan_world")

    return server


async def play_game(
    instructions: str,
    task: str,
    max_turns: int,
    model: str,
    effort: str | None = None,
    cli: str = "claude",
) -> tuple[list[ChatMessage], ModelOutput, dict]:
    """Run one game. Returns the new messages, the final output and stats."""
    server = game_server()

    # Bind a free port ourselves, so concurrent samples never collide.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    http = _QuietServer(
        uvicorn.Config(server.streamable_http_app(), log_level="warning", lifespan="on")
    )
    serving = asyncio.create_task(http.serve(sockets=[sock]))

    try:
        while not http.started:
            if serving.done():
                serving.result()  # raises the startup error
            await asyncio.sleep(0.05)

        mcp_config = {"mcpServers": {SERVER: {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}}}
        allowed = [PREFIX + name for name in ("look", "move", "dig", "scan_world")]
        args = [
            cli, "-p",
            "--model", model,
            "--system-prompt", f"{instructions}\n\n{ENDING}",
            "--output-format", "stream-json", "--verbose",
            "--mcp-config", json.dumps(mcp_config),
            "--strict-mcp-config",  # only our server
            "--allowedTools", *allowed,
            "--tools", "",  # no built-in Claude Code tools
            "--max-turns", str(max_turns + 2),  # backstop; the budget above is the real limit
            "--no-session-persistence",
            "--setting-sources", "",
            "--disable-slash-commands",
        ]
        if effort:
            args += ["--effort", effort]

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(task.encode()), timeout=GAME_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(f"claude game timed out after {GAME_TIMEOUT}s")
    finally:
        http.should_exit = True
        await serving

    events = [json.loads(line) for line in out.decode().splitlines() if line.strip()]
    result = next((e for e in events if e.get("type") == "result"), None)
    if result is None:
        raise RuntimeError(
            f"claude gave no result (exit {process.returncode}): {err.decode()[:500]!r}"
        )
    if result.get("is_error") and result.get("subtype") != "error_max_turns":
        raise RuntimeError(f"claude: {result.get('subtype')}: {str(result.get('result'))[:500]}")

    messages = to_inspect_messages(events)
    final_text = next(
        (m.text for m in reversed(messages) if isinstance(m, ChatMessageAssistant)), ""
    )
    output = ModelOutput.from_content(model, final_text)
    stats = {
        "turns_used": store_as(WorldState).turns_used,
        "hit_turn_limit": store_as(WorldState).turns_used >= max_turns,
        "claude_result": result.get("subtype"),
        "claude_num_turns": result.get("num_turns"),
        "claude_usage": result.get("usage"),
        "claude_session_id": result.get("session_id"),
    }
    return messages, output, stats


def to_inspect_messages(events: list[dict]) -> list[ChatMessage]:
    """Claude Code's stream-json events -> Inspect assistant and tool messages.

    One API message can arrive as several "assistant" events (one per content
    block) sharing a message id, so blocks are merged by id.
    """
    messages: list[ChatMessage] = []
    current_id = None
    names: dict[str, str] = {}  # tool_use id -> game tool name

    for event in events:
        message = event.get("message") or {}
        if event.get("type") == "assistant":
            texts = [b["text"] for b in message.get("content", []) if b.get("type") == "text"]
            calls = [
                ToolCall(
                    id=b["id"],
                    function=b["name"].removeprefix(PREFIX),
                    arguments=b.get("input") or {},
                )
                for b in message.get("content", [])
                if b.get("type") == "tool_use"
            ]
            for call in calls:
                names[call.id] = call.function
            if message.get("id") == current_id and isinstance(messages[-1], ChatMessageAssistant):
                last = messages[-1]
                text = "\n\n".join(filter(None, [last.text, *texts]))
                messages[-1] = ChatMessageAssistant(
                    content=text, tool_calls=(last.tool_calls or []) + calls or None
                )
            else:
                messages.append(
                    ChatMessageAssistant(content="\n\n".join(texts), tool_calls=calls or None)
                )
                current_id = message.get("id")
        elif event.get("type") == "user":
            for block in message.get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                messages.append(
                    ChatMessageTool(
                        content=str(content or ""),
                        tool_call_id=block["tool_use_id"],
                        function=names.get(block["tool_use_id"], ""),
                    )
                )
            current_id = None
    return messages
