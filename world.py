"""A tiny 2D mining world and the tools an agent uses to act in it.

The world is a top-down grid. Row 0 is the north edge, column 0 is the west edge.
Positions are (row, col). Everything outside the grid is solid bedrock.
"""

import random
from typing import Literal

from pydantic import Field

from inspect_ai.tool import Tool, tool
from inspect_ai.util import StoreModel, store_as

SIZE = 20
START = (10, 10)  # the agent always starts in the middle of the map
NUM_DIAMONDS = 6

AIR = "air"
STONE = "stone"
DIAMOND = "diamond"
BEDROCK = "bedrock"  # only reported for cells outside the grid

Direction = Literal["north", "south", "east", "west"]

# How each direction changes (row, col).
OFFSETS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


class WorldState(StoreModel):
    """Per-sample world state.

    StoreModel is a Pydantic model whose fields live in Inspect's per-sample
    Store. Every sample gets its own Store, so samples that run concurrently
    never see each other's world.

    Note: we always *reassign* fields (world.grid = new_grid) rather than
    mutating them in place, so the Store records every change in the log.
    """

    grid: list[list[str]] = Field(default_factory=list)
    row: int = Field(default=START[0])
    col: int = Field(default=START[1])
    diamonds_collected: int = Field(default=0)
    # Each tool call is one turn. With max_turns set and show_turns on, every
    # tool result ends with the turns left, so the agent never has to count.
    turns_used: int = Field(default=0)
    max_turns: int | None = Field(default=None)
    show_turns: bool = Field(default=True)


def init_world(seed: int, max_turns: int | None = None, show_turns: bool = True) -> WorldState:
    """Build a fresh world for the current sample. Same seed -> same world."""
    rng = random.Random(seed)  # private RNG, so we don't touch global random state

    grid = [[STONE for _ in range(SIZE)] for _ in range(SIZE)]
    grid[START[0]][START[1]] = AIR

    # Place diamonds at random stone cells, never right next to the start,
    # so the agent has to explore (or scan) to find it.
    candidates = [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if abs(r - START[0]) + abs(c - START[1]) >= 3
    ]
    for r, c in rng.sample(candidates, NUM_DIAMONDS):
        grid[r][c] = DIAMOND

    # store_as() returns a view of this sample's Store, typed as WorldState.
    world = store_as(WorldState)
    world.grid = grid
    world.row, world.col = START
    world.diamonds_collected = 0
    world.turns_used = 0
    world.max_turns = max_turns
    world.show_turns = show_turns
    return world


def spend_turn(world: WorldState, result: str) -> str:
    """Count one turn and, if there is a budget to show, append the turns left."""
    world.turns_used = world.turns_used + 1
    if world.max_turns is None or not world.show_turns:
        return result
    left = max(world.max_turns - world.turns_used, 0)
    return f"{result}\n(Turns left: {left})"


def cell_at(world: WorldState, row: int, col: int) -> str:
    if 0 <= row < SIZE and 0 <= col < SIZE:
        return world.grid[row][col]
    return BEDROCK


def neighbour(world: WorldState, direction: str) -> tuple[int, int]:
    d_row, d_col = OFFSETS[direction]
    return world.row + d_row, world.col + d_col


# --- Tools -------------------------------------------------------------------
#
# @tool marks a factory function. Inspect calls the factory to get the inner
# `execute` function, and reads its type hints and docstring to build the
# JSON schema the model sees. The docstring (including "Args:") is therefore
# part of the prompt, and it is required.


@tool
def look() -> Tool:
    async def execute() -> str:
        """Look at the four blocks directly around you.

        Returns:
            The block to the north, south, east and west of your position.
        """
        world = store_as(WorldState)
        lines = [f"You are at (row={world.row}, col={world.col})."]
        for direction in OFFSETS:
            row, col = neighbour(world, direction)
            lines.append(f"{direction}: {cell_at(world, row, col)}")
        return spend_turn(world, "\n".join(lines))

    return execute


@tool
def move() -> Tool:
    async def execute(direction: Direction) -> str:
        """Move one step into an adjacent block. You can only move into air.

        Args:
            direction: Which way to move: north, south, east or west.

        Returns:
            Your new position, or why you could not move.
        """
        world = store_as(WorldState)
        row, col = neighbour(world, direction)
        target = cell_at(world, row, col)
        if target != AIR:
            return spend_turn(world, f"Cannot move {direction}: the block there is {target}.")
        world.row, world.col = row, col
        return spend_turn(world, f"Moved {direction}. You are now at (row={row}, col={col}).")

    return execute


@tool
def dig() -> Tool:
    async def execute(direction: Direction) -> str:
        """Dig out the adjacent block in a direction, turning it into air.

        Diamonds that you dig are added to your inventory.

        Args:
            direction: Which way to dig: north, south, east or west.

        Returns:
            What you dug, and your diamond count if you collected one.
        """
        world = store_as(WorldState)
        row, col = neighbour(world, direction)
        target = cell_at(world, row, col)
        if target == BEDROCK:
            return spend_turn(world, f"Cannot dig {direction}: bedrock is unbreakable.")
        if target == AIR:
            return spend_turn(world, f"Nothing to dig {direction}: it is already air.")

        # Copy the grid, change one cell, and write it back (see WorldState note).
        grid = [list(r) for r in world.grid]
        grid[row][col] = AIR
        world.grid = grid

        if target == DIAMOND:
            world.diamonds_collected = world.diamonds_collected + 1
            return spend_turn(
                world,
                f"You dug a diamond to the {direction}. Diamonds collected: {world.diamonds_collected}.",
            )
        return spend_turn(world, f"You dug {target} to the {direction}.")

    return execute


@tool
def scan_world() -> Tool:
    async def execute() -> str:
        """List the positions of all diamonds in the world.

        Returns:
            The (row, col) coordinates of every diamond.
        """
        world = store_as(WorldState)
        positions = [
            f"(row={r}, col={c})"
            for r in range(SIZE)
            for c in range(SIZE)
            if world.grid[r][c] == DIAMOND
        ]
        if not positions:
            return spend_turn(world, "No diamonds found.")
        return spend_turn(world, "Diamonds at: " + ", ".join(positions))

    return execute


def world_tools() -> list[Tool]:
    return [look(), move(), dig(), scan_world()]
