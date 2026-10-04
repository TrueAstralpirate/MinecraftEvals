"""A scripted depth-first-search player, so we can run the eval without an LLM.

Usage:
    uv run python bot.py
    uv run inspect view                          # transcript
    uv run python replay.py && open replay.html  # step through the game

Inspect's built-in "mockllm" model accepts a Python function in place of a real
model. Inspect calls it with the conversation each turn, and it returns the
next "model output" (here: one tool call). Everything else is the real eval
pipeline: react() loop, tools, turn limit, logs.

The bot is stateless, like an LLM: each turn it re-reads the conversation.
It is an honest player: it only uses look() and never calls scan_world().
"""

import re

from inspect_ai import eval
from inspect_ai.model import ChatMessage, ChatMessageTool, ModelOutput, get_model

from task import xray
from world import OFFSETS

MODEL = "mockllm/model"


def call(tool: str, reasoning: str, **arguments) -> ModelOutput:
    # A fake model response: some text plus one tool call.
    return ModelOutput.for_tool_call(MODEL, tool, arguments, content=reasoning)


def parse_look(text: str) -> tuple[tuple[int, int], dict[str, str]]:
    """Turn look()'s output into ((row, col), {"north": "stone", ...})."""
    match = re.search(r"row=(\d+), col=(\d+)", text)
    position = (int(match.group(1)), int(match.group(2)))
    neighbours = dict(re.findall(r"^(north|south|east|west): (\w+)$", text, re.MULTILINE))
    return position, neighbours


def direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    for direction, (d_row, d_col) in OFFSETS.items():
        if (start[0] + d_row, start[1] + d_col) == end:
            return direction
    raise ValueError(f"{end} is not next to {start}")


def dfs_bot(messages: list[ChatMessage], tools, tool_choice, config) -> ModelOutput:
    tool_results = [m for m in messages if isinstance(m, ChatMessageTool)]
    last = tool_results[-1] if tool_results else None

    # Start of the game, or we just moved: look around first.
    if last is None or last.function == "move":
        return call("look", "Let me look around.")

    if "Diamonds collected" in last.text:
        return call("submit", "Got a diamond, done.", answer="I collected a diamond by exploring.")

    # We just dug stone. The dig direction is in the text: "You dug stone to the east."
    if last.function == "dig":
        # (The first line; the second is the "(Turns left: N)" counter.)
        direction = last.text.splitlines()[0].rstrip(".").split()[-1]
        return call("move", f"Stepping into the tunnel I just dug to the {direction}.", direction=direction)

    # Otherwise the last tool was look(): decide where to go next.
    position, neighbours = parse_look(last.text)

    # Every position we have stood on, in order (we look once after every move).
    history = [parse_look(m.text)[0] for m in tool_results if m.function == "look"]
    visited = set(history)

    # 1. Diamond right next to us? Dig it.
    for direction, block in neighbours.items():
        if block == "diamond":
            return call("dig", f"I can see a diamond to the {direction}!", direction=direction)

    # 2. Go deeper: first unvisited neighbour we can get into.
    for direction, block in neighbours.items():
        d_row, d_col = OFFSETS[direction]
        target = (position[0] + d_row, position[1] + d_col)
        if target in visited or block == "bedrock":
            continue
        if block == "stone":
            return call("dig", f"Nothing here. Tunnelling {direction} into new stone.", direction=direction)
        return call("move", f"Walking {direction} into open air.", direction=direction)

    # 3. Dead end: backtrack. Rebuild the DFS stack from the path we walked:
    #    stepping back onto the previous cell pops, stepping somewhere new pushes.
    stack: list[tuple[int, int]] = []
    for pos in history:
        if len(stack) >= 2 and pos == stack[-2]:
            stack.pop()
        elif not stack or pos != stack[-1]:
            stack.append(pos)

    if len(stack) < 2:
        return call("submit", "Explored everything I can reach.", answer="I could not find any diamonds.")
    back = direction_to(position, stack[-2])
    return call("move", f"Dead end, backtracking {back}.", direction=back)


if __name__ == "__main__":
    logs = eval(xray(), model=get_model(MODEL, custom_outputs=dfs_bot))
    for sample in logs[0].samples or []:
        diamonds = sample.store.get("WorldState:diamonds_collected", 0)
        print(f"{sample.id}: diamonds collected = {diamonds}, hit turn limit = {sample.metadata['hit_turn_limit']}")
