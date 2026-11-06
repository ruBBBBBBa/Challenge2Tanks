"""
PFSP (prioritized fictitious self-play) callback.

Tracks the win rate against every opponent from info["opponent"] ("pool[<spec>]") with a
Beta(1,1) prior over the last `window` games, and every `every` episodes re-weights the
league part of the pool:
    variance -> p(1-p)   (even opponents; early training)
    hard     -> (1-p)^2  (opponents that beat us; late training)
    off      -> uniform  (only logs the win rates)
The fixed part (bots, baselines) keeps its share (1 - league_share).
"""
from collections import defaultdict, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

SCORES = {
    "variance": lambda p: p * (1 - p),
    "hard": lambda p: (1 - p) ** 2,
    "off": lambda p: np.ones_like(p),
}


def mix_weights(fixed, league, league_share, scores=None, min_weight=0.02):
    """{spec: weight} summing to 1: fixed specs scaled to 1-league_share, league specs by score."""
    if not league:
        total = sum(fixed.values())
        return {s: w / total for s, w in fixed.items()}
    if not fixed:
        league_share = 1.0
    scores = np.ones(len(league)) if scores is None else np.asarray(scores, dtype=float)
    lw = scores / scores.sum() if scores.sum() > 0 else np.full(len(league), 1 / len(league))
    lw = np.maximum(lw * league_share, min_weight)   # min_weight is a share of the whole pool
    lw = lw / lw.sum() * league_share
    total = sum(fixed.values())
    out = {s: (1 - league_share) * w / total for s, w in fixed.items()}
    for s, w in zip(league, lw):
        out[s] = out.get(s, 0.0) + float(w)
    return out


class PFSPCallback(BaseCallback):
    def __init__(self, fixed, league, league_share=0.5, mode="variance", every=200, window=100,
                 min_weight=0.02, verbose=1):
        super().__init__(verbose)
        self.fixed, self.league = dict(fixed), list(league)
        self.league_share, self.mode, self.every = league_share, mode, every
        self.min_weight = min_weight
        self.results = defaultdict(lambda: deque(maxlen=window))
        self.episodes = 0

    def win_rate(self, spec):
        r = self.results[spec]
        return (sum(r) + 1) / (len(r) + 2)    # Beta(1,1) posterior mean

    def weights(self):
        scores = SCORES[self.mode](np.array([self.win_rate(s) for s in self.league]))
        return mix_weights(self.fixed, self.league, self.league_share, scores, self.min_weight)

    def _on_training_start(self):
        self.training_env.env_method("set_pool_weights", mix_weights(self.fixed, self.league, self.league_share))

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            name = info.get("opponent", "") if done else ""
            if done and "result" in info and name.startswith("pool["):
                self.results[name[5:-1]].append(info["is_success"])
                self.episodes += 1
                if self.episodes % self.every == 0:
                    self._update()
        return True

    def _update(self):
        for spec in self.results:
            self.logger.record(f"pfsp/wr/{spec}", self.win_rate(spec))
        if self.league and self.mode != "off":
            w = self.weights()
            self.training_env.env_method("set_pool_weights", w)
            if self.verbose:
                top = sorted(((v, s) for s, v in w.items() if s in self.league), reverse=True)[:3]
                print("[pfsp] top league weights:", ", ".join(f"{s}={v:.3f}" for v, s in top))


if __name__ == "__main__":
    w = mix_weights({"basic": 1, "baseline:hard": 3}, ["a", "b", "c"], 0.5, [0.25, 0.0, 0.0])
    assert abs(sum(w.values()) - 1) < 1e-9 and abs(w["baseline:hard"] - 0.375) < 1e-9
    assert w["b"] > 0 and w["a"] > w["b"], w
    assert mix_weights({"basic": 2}, [], 0.5) == {"basic": 1.0}
    cb = PFSPCallback({"basic": 1}, ["a", "b"], mode="hard")
    cb.results["a"].extend([1] * 20)
    cb.results["b"].extend([0] * 20)
    w = cb.weights()
    assert w["b"] > w["a"] and abs(w["basic"] - 0.5) < 1e-9, w
    print("pfsp OK", w)
