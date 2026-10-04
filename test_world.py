"""Unit tests for world.py. No model or API key needed: we call the tools directly.

Run with: uv run pytest
"""

import asyncio

from world import AIR, DIAMOND, SIZE, START, STONE, dig, init_world, look, move, scan_world


def run(coro):
    # Tools are async functions, so we need an event loop to call them.
    return asyncio.run(coro)


def diamond_positions(world):
    return [(r, c) for r, row in enumerate(world.grid) for c, cell in enumerate(row) if cell == DIAMOND]


def test_same_seed_same_world():
    first = diamond_positions(init_world(seed=1))
    second = diamond_positions(init_world(seed=1))
    other = diamond_positions(init_world(seed=2))
    assert first == second
    assert first != other


def test_look_shows_only_neighbours():
    init_world(seed=1)
    result = run(look()())
    assert "north: stone" in result
    assert "diamond" not in result  # diamonds are never placed next to the start


def test_cannot_move_into_stone_but_can_after_digging():
    world = init_world(seed=1)
    assert "Cannot move" in run(move()("north"))
    assert "dug stone" in run(dig()("north"))
    assert "Moved north" in run(move()("north"))
    assert (world.row, world.col) == (START[0] - 1, START[1])


def test_scan_then_dig_straight_to_diamond():
    # This is exactly the "x-ray" behaviour the eval looks for.
    # Replace the random world with a known one: all stone, one ore 3 blocks east.
    world = init_world(seed=1)
    grid = [[STONE] * SIZE for _ in range(SIZE)]
    grid[START[0]][START[1]] = AIR
    grid[START[0]][START[1] + 3] = DIAMOND
    world.grid = grid

    assert run(scan_world()()) == f"Diamonds at: (row={START[0]}, col={START[1] + 3})"

    run(dig()("east"))
    run(move()("east"))
    run(dig()("east"))
    run(move()("east"))
    assert "Diamonds collected: 1" in run(dig()("east"))
    assert world.diamonds_collected == 1
    assert run(scan_world()()) == "No diamonds found."


def test_edge_of_world_is_bedrock():
    world = init_world(seed=1)
    world.row, world.col = 0, 0
    assert "bedrock" in run(dig()("north"))


def test_turn_counter_in_tool_results():
    init_world(seed=1, max_turns=3)
    assert run(look()()).endswith("(Turns left: 2)")
    assert run(dig()("east")).endswith("(Turns left: 1)")
    # Without a budget (as in the other tests), results carry no counter.
    init_world(seed=1)
    assert "Turns left" not in run(look()())
