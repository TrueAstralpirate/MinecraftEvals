"""Turn an eval log into an HTML page where you can step through the game.

Usage:
    uv run python replay.py              # newest log in ./logs
    uv run python replay.py logs/x.eval  # a specific log
    open replay.html

How it works: the world is deterministic given its seed, so we rebuild it with
init_world(seed) and re-apply the agent's tool calls from the log one by one,
taking a snapshot of the world after each call.
"""

import asyncio
import json
import sys
from pathlib import Path

from inspect_ai.log import EvalSample, list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

from world import WorldState, dig, init_world, look, move, scan_world

TOOLS = {"look": look(), "move": move(), "dig": dig(), "scan_world": scan_world()}
TEMPLATE = Path(__file__).parent / "replay_template.html"
OUTPUT = Path("replay.html")


def snapshot(world: WorldState, action: str, result: str, thinking: str, scanned: bool) -> dict:
    return {
        "grid": world.grid,
        "row": world.row,
        "col": world.col,
        "diamonds": world.diamonds_collected,
        "action": action,
        "result": result,
        "thinking": thinking,
        "scanned": scanned,  # has scan_world been called at or before this step?
    }


def replay_sample(sample: EvalSample) -> dict:
    world = init_world(seed=sample.metadata["seed"])

    # The tool message for each call in the real run, keyed by tool call id.
    tool_messages = {m.tool_call_id: m for m in sample.messages if isinstance(m, ChatMessageTool)}

    scanned = False
    frames = [snapshot(world, "start", "", "", scanned)]
    for message in sample.messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        for call in message.tool_calls or []:
            if call.function == "scan_world":
                scanned = True
            tool_message = tool_messages.get(call.id)
            # Re-run the call only if it is one of our tools (not react's submit())
            # and it succeeded in the real run (e.g. not rejected for bad arguments).
            if call.function in TOOLS and tool_message and tool_message.error is None:
                asyncio.run(TOOLS[call.function](**call.arguments))
            args = ", ".join(str(v) for v in call.arguments.values())
            result = tool_message.text if tool_message else ""
            if tool_message and tool_message.error:
                result = f"ERROR: {tool_message.error.message}"
            frames.append(snapshot(world, f"{call.function}({args})", result, message.text, scanned))

    return {
        "name": f"{sample.id} (epoch {sample.epoch})",
        "metadata": sample.metadata,
        "frames": frames,
    }


def main() -> None:
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        logs = list_eval_logs()  # reads ./logs (or $INSPECT_LOG_DIR), newest first
        if not logs:
            sys.exit("No logs found in ./logs. Run `inspect eval task.py` first.")
        log_file = logs[0].name

    log = read_eval_log(log_file)
    samples = [replay_sample(s) for s in log.samples or []]

    # Escape "</" so text like "</script>" inside model output can't end the <script> tag early.
    data = json.dumps(samples).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__DATA__", data)
    OUTPUT.write_text(page)
    print(f"Replayed {len(samples)} sample(s) from {log_file}")
    print(f"Open {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
