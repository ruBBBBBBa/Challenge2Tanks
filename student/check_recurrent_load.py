"""
Load check of a recurrent model with the official loader (PC2).

    python student/check_recurrent_load.py [model.zip]      (default models/D2b-distill.zip)

Prints the versions, the class load_sb3_model returns (PPO is tried first and accepts a RecurrentPPO zip:
the object is PPO but its policy is RecurrentActorCriticPolicy, which keeps the LSTM state), the LSTM state
after one predict, and one game vs hard with ModelAgent twice (determinism) and with an explicit
RecurrentPPO.load (same actions). Run it in every clean environment (e.g. Python 3.8 + SB3/sb3-contrib 2.1.0).
"""
import os
import sys

import numpy as np
import stable_baselines3 as sb3
import sb3_contrib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from tanks.agents.model_agent import ModelAgent, load_sb3_model  # noqa: E402
from tanks.maps import list_maps  # noqa: E402
from tanks.match import play_match  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "models/D2b-distill.zip"
print("python", sys.version.split()[0], "sb3", sb3.__version__, "sb3-contrib", sb3_contrib.__version__)
m = load_sb3_model(path)
print("load_sb3_model ->", type(m).__name__, "| policy", type(m.policy).__name__)

# LSTM state through ModelAgent.act
a = ModelAgent(path, deterministic=True)
a._ensure_loaded()
st = a._rnn_state
obs = np.zeros(77, dtype=np.float32)
_, s1 = a.model.predict(obs, state=None, episode_start=np.ones(1, bool), deterministic=True)
print("state after 1st predict:", None if s1 is None else (type(s1).__name__, [x.shape for x in s1]))

# Same actions as an explicit RecurrentPPO, over a whole game, and deterministic
ref = RecurrentPPO.load(path, device="cpu")


class Explicit(ModelAgent):
    def _ensure_loaded(self):
        self.model = self.model or ref


mp = list_maps()[0]
r1 = play_match(ModelAgent(path), "baseline:hard", mp, seed=1)
r2 = play_match(ModelAgent(path), "baseline:hard", mp, seed=1)
r3 = play_match(Explicit(path), "baseline:hard", mp, seed=1)
key = lambda r: (r["winner"], r["steps"], tuple(r["damage"]) if hasattr(r["damage"], "__iter__") else r["damage"])  # noqa: E731
print("vs hard on", mp, "| official:", key(r1), "| repeat:", key(r2), "| explicit RecurrentPPO:", key(r3))
print("deterministic:", key(r1) == key(r2), "| same as RecurrentPPO:", key(r1) == key(r3))
