# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Run one game between two agents (used by play_match, evaluate, tournament).
Also saves / loads replays (JSON with map + config + actions: games are deterministic).
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Union

import numpy as np

from tanks.agents.base import Agent, load_agent
from tanks.game import GameConfig, GameMap, TanksGame
from tanks.maps import load_map

REPLAY_VERSION = 1


def play_match(agent0: Union[str, Agent], agent1: Union[str, Agent],
               game_map: Union[str, GameMap] = "random", config: Optional[GameConfig] = None,
               seed: Optional[int] = None, render: bool = False, fps: float = 8,
               record_frames: bool = False, renderer=None, names: Optional[List[str]] = None) -> dict:
    """Play one full game. Returns a result dict (winner is 0, 1 or None for a draw)."""
    rng = np.random.default_rng(seed)
    agents = [load_agent(agent0), load_agent(agent1)]
    names = names or [a.name for a in agents]
    if isinstance(game_map, str):
        game_map = load_map(game_map, rng)
    game = TanksGame(game_map, config)
    for a in agents:
        a.reset(int(rng.integers(2**31 - 1)))

    frames = []
    own_renderer = False
    if (render or record_frames) and renderer is None:
        from tanks.render import Renderer
        renderer, own_renderer = Renderer(), True

    def draw():
        surf = renderer.draw(game, names)
        if record_frames:
            frames.append(renderer.to_array(surf))
        if render and not renderer.show(surf, fps):
            raise KeyboardInterrupt("window closed")

    think_time = [0.0, 0.0]
    if renderer is not None:
        draw()
    while not game.done:
        actions = []
        for p in (0, 1):
            t0 = time.perf_counter()
            actions.append(agents[p].act(game.get_state(p)))
            think_time[p] += time.perf_counter() - t0
        game.step(actions)
        if renderer is not None:
            draw()
    if renderer is not None and render:
        for _ in range(int(fps)):   # hold the last frame ~1 s
            renderer.show(renderer.draw(game, names), fps)
    if own_renderer and render:
        renderer.close()

    t0, t1 = game.tanks
    return {
        "winner": game.winner,
        "end_reason": game.end_reason,
        "steps": game.step_count,
        "hp": [t0.hp, t1.hp],
        "damage": [t0.damage_dealt, t1.damage_dealt],
        "shots": [t0.shots_fired, t1.shots_fired],
        "names": names,
        "map_name": game_map.name,
        "map": game_map.to_ascii(),
        "config": game.cfg.to_dict(),
        "actions": game.action_history,
        "ms_per_action": [1000 * tt / max(1, game.step_count) for tt in think_time],
        "frames": frames,
    }


def save_replay(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {k: v for k, v in result.items() if k != "frames"}
    data["version"] = REPLAY_VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_replay(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def replay_game(data: dict):
    """Yield the TanksGame after each step of a replay (first yield = initial state)."""
    game_map = GameMap.from_ascii(data["map"], name=data.get("map_name", "replay"))
    game = TanksGame(game_map, GameConfig.from_dict(data["config"]))
    yield game
    for acts in data["actions"]:
        game.step(acts)
        yield game
