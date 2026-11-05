# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Wraps a Stable-Baselines3 model (.zip) as an Agent.

The model receives the official observation `observations.vector_obs(state)`,
exactly as in the tournament. Supported: PPO, A2C, DQN and (if installed)
sb3-contrib algorithms such as RecurrentPPO, QRDQN, TRPO, ARS, MaskablePPO.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from tanks.agents.base import Agent
from tanks.game import N_ACTIONS, NOOP
from tanks.observations import OBS_SIZE, vector_obs

# Objects that are pickled inside the zip and are not needed for inference.
# Replacing them avoids version problems (and some unpickling).
_CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "clip_range_vf": None,
    "exploration_schedule": lambda _: 0.0,
}


def _algorithms():
    from stable_baselines3 import A2C, DQN, PPO
    algos = [PPO, A2C, DQN]
    try:
        import sb3_contrib as c
        algos += [c.RecurrentPPO, c.MaskablePPO, c.QRDQN, c.TRPO, c.ARS]
    except ImportError:
        pass
    return algos


def load_sb3_model(path: str, device: str = "cpu"):
    """Load an SB3 model without knowing its algorithm. Checks the spaces."""
    errors = []
    for algo in _algorithms():
        try:
            model = algo.load(path, device=device, custom_objects=_CUSTOM_OBJECTS)
            break
        except Exception as e:  # wrong algorithm class -> try the next one
            errors.append(f"{algo.__name__}: {type(e).__name__}: {e}")
    else:
        raise RuntimeError(f"Could not load {path} with any algorithm:\n  " + "\n  ".join(errors))

    obs_shape = getattr(model.observation_space, "shape", None)
    n = getattr(model.action_space, "n", None)
    if obs_shape != (OBS_SIZE,):
        raise ValueError(f"{path}: observation shape {obs_shape}, expected ({OBS_SIZE},) "
                         f"(models must use the official observation)")
    if n != N_ACTIONS:
        raise ValueError(f"{path}: action space {model.action_space}, expected Discrete({N_ACTIONS})")
    return model


class ModelAgent(Agent):
    def __init__(self, path: str, deterministic: bool = True, device: str = "cpu",
                 name: Optional[str] = None, lazy: bool = True):
        self.path = path
        self.deterministic = deterministic
        self.device = device
        self.name = name or os.path.relpath(path)
        self.model = None
        self._rnn_state = None
        self._episode_start = np.ones((1,), dtype=bool)
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is None:
            self.model = load_sb3_model(self.path, self.device)

    def reset(self, seed: Optional[int] = None) -> None:
        self._rnn_state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def act(self, state: dict) -> int:
        self._ensure_loaded()
        obs = vector_obs(state)
        action, self._rnn_state = self.model.predict(
            obs, state=self._rnn_state, episode_start=self._episode_start,
            deterministic=self.deterministic)
        self._episode_start = np.zeros((1,), dtype=bool)
        action = int(np.asarray(action).item())
        return action if 0 <= action < N_ACTIONS else NOOP

    def __getstate__(self):
        # do not pickle the torch model (e.g. when sent to SubprocVecEnv workers)
        d = dict(self.__dict__)
        d["model"] = None
        return d
