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

from tanks.agents import Agent
from tanks.agents.heuristics import ParamHeuristic, enemy_in_sights, turn_towards  # noqa: F401
from tanks.game import (BACKWARD, FIRE, FORWARD, NOOP, TURN_LEFT, TURN_RIGHT,  # noqa: F401
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


# TODO: write your own bots here.


# "my:<name>" -> MY_BOTS[name]()   (use classes or functools.partial, not lambdas)
MY_BOTS = {
    "camper": CamperBot,
    "wanderer": WandererBot,
    # heuristic with your own parameters (same as the spec "param:...")
    "sniper": functools.partial(ParamHeuristic, aggression=0.3, caution=0.8, trigger=1.0, noise=0.0, flank=0.9),
}
