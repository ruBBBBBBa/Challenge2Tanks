# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
TanksEnv: single-agent Gymnasium environment. The opponent lives INSIDE the env,
so for Stable-Baselines3 this is a normal environment (like the Snake one).

    env = TanksEnv(opponent="basic")                                 # heuristic
    env = TanksEnv(opponent="baseline:easy")                         # teacher model
    env = TanksEnv(opponent="model:models/my_v1.zip")                # your own frozen model
    env = TanksEnv(opponent={"basic": 0.5, "aggressive": 0.5})       # mixture, one per episode

Observation: official vector (tanks/observations.py, Box(77,)) -- the tournament uses the same one.
Action:      Discrete(6): noop, forward, backward, turn_left, turn_right, fire.
Reward:      rewards.default_reward unless you pass reward_fn (see tanks/rewards.py and student/my_rewards.py).
Episode end: terminated when a tank is destroyed, truncated at config.max_steps
             (the game is still decided by tie-break, see info["result"]).
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tanks.agents.base import Agent, load_agent
from tanks.game import ACTION_NAMES, N_ACTIONS, GameConfig, TanksGame, valid_actions
from tanks.maps import list_maps, load_map
from tanks.observations import OBS_SIZE, vector_obs
from tanks.rewards import default_reward

OpponentSpec = Union[str, Dict[str, float], Agent]

# VecEnv.env_method("set_opponent", ...) goes through the Monitor wrapper; gymnasium 0.29
# warns about that attribute forwarding. It is harmless (and gone in newer SB3), so silence it.
warnings.filterwarnings("ignore", message=r".*to get variables from other wrappers is deprecated.*")


class TanksEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self,
                 opponent: OpponentSpec = "basic",
                 maps: Optional[Sequence[str]] = None,
                 config: Optional[Union[GameConfig, dict]] = None,
                 reward_fn: Optional[Callable[[dict, dict, dict], float]] = None,
                 learner_side: Optional[int] = None,
                 render_mode: Optional[str] = None):
        """
        opponent      agent spec (see tanks/agents/base.py) -- string, dict (pool) or Agent
        maps          list of map names; "random" = generated symmetric map.
                      Default: all official maps + "random" with 50% probability.
        config        GameConfig or dict (train with other rules if you like;
                      the tournament uses the defaults)
        reward_fn     reward_fn(prev_state, state, events) -> float
        learner_side  0, 1, or None = random side every episode (recommended)
        render_mode   None, "human" or "rgb_array"
        """
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        if maps is None:
            official = list_maps()
            maps = official + ["random"] * max(1, len(official))
        self.maps: List[str] = list(maps)
        self.config = GameConfig.from_dict(config) if not isinstance(config, GameConfig) else config
        self.reward_fn = reward_fn or default_reward
        self.learner_side = learner_side
        self.render_mode = render_mode

        self.opponent: Agent = load_agent(opponent)
        self.game: Optional[TanksGame] = None
        self.side = 0
        self._state = None
        self._renderer = None

    # ------------------------------------------------------------------ API
    def set_opponent(self, opponent: OpponentSpec) -> None:
        """Change the opponent (takes effect at the next reset). Used by tanks/curriculum.py."""
        self.opponent = load_agent(opponent)

    def get_opponent_name(self) -> str:
        return self.opponent.name

    def valid_actions(self) -> np.ndarray:
        """Boolean mask of non-useless actions for the learner (see game.valid_actions)."""
        return valid_actions(self._state)

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        map_name = (options or {}).get("map") or self.maps[int(rng.integers(len(self.maps)))]
        self.game = TanksGame(load_map(map_name, rng), self.config)
        self.side = self.learner_side if self.learner_side is not None else int(rng.integers(2))
        self.opponent.reset(int(rng.integers(2**31 - 1)))
        self._state = self.game.get_state(self.side)
        if self.render_mode == "human":
            self.render()
        info = {"map": self.game.map.name, "side": self.side, "opponent": self.opponent.name}
        return vector_obs(self._state), info

    def step(self, action):
        opp = 1 - self.side
        opp_action = self.opponent.act(self.game.get_state(opp))
        actions = [0, 0]
        actions[self.side] = int(action)
        actions[opp] = int(opp_action)
        events = self.game.step(actions)[self.side]

        prev_state = self._state
        self._state = self.game.get_state(self.side)
        reward = float(self.reward_fn(prev_state, self._state, events))

        terminated = truncated = False
        info = {"events": events}
        if self.game.done:
            if self.game.end_reason == "timeout":
                truncated = True
            else:
                terminated = True
            me, en = self.game.tanks[self.side], self.game.tanks[opp]
            info.update({
                "result": events["result"],
                "end_reason": events["end_reason"],
                "is_success": float(events["result"] == "win"),
                "hp": me.hp,
                "enemy_hp": en.hp,
                "damage_dealt": me.damage_dealt,
                "opponent": self.opponent.name,
                "map": self.game.map.name,
            })
        if self.render_mode == "human":
            self.render()
        return vector_obs(self._state), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.game is None:
            return None
        if self._renderer is None:
            from tanks.render import Renderer
            self._renderer = Renderer()
        names = ["learner", self.opponent.name] if self.side == 0 else [self.opponent.name, "learner"]
        surf = self._renderer.draw(self.game, names)
        if self.render_mode == "human":
            self._renderer.show(surf, self.metadata["render_fps"])
            return None
        return self._renderer.to_array(surf)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # helper for debugging
    @staticmethod
    def action_name(a: int) -> str:
        return ACTION_NAMES[int(a)]
