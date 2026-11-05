"""
YOUR own opponents (bots) to train against.

A bot is a class with `reset(seed)` and `act(state) -> action` (subclass tanks.agents.Agent).
Register it in MY_BOTS and use it anywhere with the spec "my:<name>":

    TanksEnv(opponent="my:camper")
    TanksEnv(opponent={"basic": 0.5, "my:camper": 0.5})     # in a mixture
    python tools/play_match.py --p0 my:camper --p1 basic    # watch it

Why? If your agent has a weakness (e.g. it never looks for an enemy that camps behind
bricks), write a bot that exploits it and add it to your training mixture.

Bots are for TRAINING only: the tournament uses only your model.zip.
"""
import functools

import numpy as np

from tanks.agents import Agent, load_agent
from tanks.agents.heuristics import (PRESETS, ParamHeuristic, bfs_first_step, enemy_in_sights,  # noqa: F401
                                     turn_towards)
from tanks.game import (BACKWARD, BUSH, DRIVABLE, FIRE, FORWARD, NOOP, TURN_LEFT, TURN_RIGHT,  # noqa: F401
                        clear_line, direction_to, valid_actions)

# Actions: NOOP=0, FORWARD=1, BACKWARD=2, TURN_LEFT=3, TURN_RIGHT=4, FIRE=5
# Directions: UP=0, RIGHT=1, DOWN=2, LEFT=3 (state["me"]["dir"])


class CamperBot(Agent):
    """Example: never moves. Turns towards the enemy when aligned and fires when it can."""

    name = "camper"

    def __init__(self):
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        me, en = state["me"], state["enemy"]
        if not en["visible"]:
            return NOOP
        if enemy_in_sights(state):
            return FIRE if me["cooldown"] == 0 else NOOP
        d = direction_to(me["x"], me["y"], en["x"], en["y"])
        if d is not None and clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"]):
            return turn_towards(me["dir"], d)
        return NOOP


class WandererBot(Agent):
    """Example: drives around randomly (only valid moves) and fires when the enemy is in its sights."""

    name = "wanderer"

    def __init__(self):
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        if enemy_in_sights(state) and state["me"]["cooldown"] == 0:
            return FIRE
        mask = valid_actions(state)
        if mask[FORWARD] and self.rng.random() < 0.7:
            return FORWARD
        return int(self.rng.choice([TURN_LEFT, TURN_RIGHT]))


def aim_or_fire(state):
    """FIRE if the enemy is in my sights and I am loaded; turn towards it if aligned with a clear line."""
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return None
    if enemy_in_sights(state):
        return FIRE if me["cooldown"] == 0 else NOOP
    d = direction_to(me["x"], me["y"], en["x"], en["y"])
    if d is not None and clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"]):
        return turn_towards(me["dir"], d)
    return None


class _HideBot(Agent):
    """Shared logic: hunt with a ParamHeuristic, hide in the nearest bush (or retreat) when told to."""

    hunter = "basic"

    def __init__(self):
        self.h = ParamHeuristic(**PRESETS[self.hunter])

    def reset(self, seed=None):
        self.h.reset(seed)

    def hide(self, state):
        me, tiles = state["me"], state["tiles"]
        pos = (me["x"], me["y"])
        grid = tiles.tolist()
        res = bfs_first_step(tiles, pos, lambda c: grid[c[1]][c[0]] == BUSH, self.h._passable(state))
        if res is not None:
            d, _ = res
            return NOOP if d is None else self.h._move(state, d)
        last = self.h.last_seen
        return self.h._retreat(state, last) if last is not None else NOOP   # no bush on this map

    def act(self, state):
        en = state["enemy"]
        if en["visible"]:
            self.h.last_seen = (en["x"], en["y"])
        a = aim_or_fire(state)
        if a is not None:
            return a
        return self.hide(state) if self.hiding(state) else self.h.act(state)


class BushCamperBot(_HideBot):
    """Goes to the nearest bush and waits there, firing when the enemy enters its line.
    Leaves to hunt only in the second half of the game and only if it is not ahead in HP."""

    name = "bushcamper"

    def hiding(self, state):
        ahead = state["me"]["hp"] > state["enemy"]["hp"]
        return ahead or state["step"] < state["max_steps"] // 2


class HitAndHideBot(_HideBot):
    """Hunts until it is ahead in HP, then hides (bush or far away) until the end."""

    name = "hitandhide"

    def hiding(self, state):
        return state["me"]["hp"] > state["enemy"]["hp"]


def stochastic_model(spec):
    """"my:stoch:<model spec>" -> the same model sampling its actions (deterministic=False)."""
    return load_agent(spec, deterministic=False)


class _BotDict(dict):
    """MY_BOTS that also accepts "stoch:<spec>" (e.g. my:stoch:baseline:hard, my:stoch:models/A-g1.zip)."""

    def __contains__(self, name):
        return name.startswith("stoch:") or super().__contains__(name)

    def __getitem__(self, name):
        if name.startswith("stoch:"):
            return functools.partial(stochastic_model, name[len("stoch:"):])
        return super().__getitem__(name)


# "my:<name>" -> MY_BOTS[name]()   (use classes or functools.partial, not lambdas)
MY_BOTS = _BotDict({
    "camper": CamperBot,
    "wanderer": WandererBot,
    # heuristic with your own parameters (same as the spec "param:...")
    "sniper": functools.partial(ParamHeuristic, aggression=0.3, caution=0.8, trigger=1.0, noise=0.0, flank=0.9),
    # ponytail: rusher = ParamHeuristic with no dodging/flanking/noise; write a BFS bot if it is too cautious
    "rusher": functools.partial(ParamHeuristic, aggression=1.0, caution=0.0, trigger=1.0, noise=0.0, flank=0.0),
    "bushcamper": BushCamperBot,
    "hitandhide": HitAndHideBot,
})
