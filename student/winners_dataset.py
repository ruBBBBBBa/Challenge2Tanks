"""
Imitation data from a league: (obs, action) sequences of the WINNING side, only from games won by
the N best models of the ranking. Input: a league run with --save-all-replays.

    python student/winners_dataset.py --league results/league_2026-11-07_0900_liga2.json --top 3 \
        --out models/checkpoints_D-g2/winners.npz
    python student/distill.py train --data models/checkpoints_D-g2/winners.npz --out models/D-g2-init.zip

Output = the format of `distill.py gen`: obs (L, E, 77) float32, probs (L, E, 6) float32 (one-hot of the
action taken), rew (L, E) float32 (tanks default_reward, winner's view), done (L, E) bool. Episodes are
packed back to back in E streams; the tail of each stream is zero padding with done=False, which
distill.py's returns_and_mask() already masks for the value loss (see BITACORA.md for the actor loss).
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from student.league import replay_events  # noqa: E402
from tanks.game import GameMap  # noqa: E402
from tanks.observations import OBS_SIZE, vector_obs  # noqa: E402
from tanks.rewards import default_reward  # noqa: E402


def episode(rep, winner):
    """Re-simulate one replay; returns obs (T, 77), actions (T,), rewards (T,) of player `winner`."""
    game_map = GameMap.from_ascii(rep["map"], name=rep.get("map_name", "replay"))
    obs, acts, rews = [], [], []
    prev = None
    for t, (game, ev) in enumerate(replay_events(game_map, rep["config"], rep["actions"])):
        state = game.get_state(winner)
        if ev is not None:
            rews.append(default_reward(prev, state, ev[winner]))
        if t < len(rep["actions"]):
            obs.append(vector_obs(state))
            acts.append(rep["actions"][t][winner])
        prev = state
    assert game.winner == rep["winner"], "replay diverged"
    return np.array(obs, np.float32), np.array(acts, np.int64), np.array(rews, np.float32)


def predict(cache, spec, obs):
    """Deterministic actions of a model (MLP or LSTM) over one episode of observations."""
    from tanks.agents.model_agent import load_sb3_model
    path = spec[len("model:"):] if spec.startswith("model:") else spec
    if path not in cache:
        cache[path] = load_sb3_model(path)
    state, start, out = None, np.ones((1,), bool), []
    for o in obs:
        a, state = cache[path].predict(o, state=state, episode_start=start, deterministic=True)
        start = np.zeros((1,), bool)
        out.append(int(np.asarray(a).item()))
    return np.array(out)


def pack(episodes, n_streams):
    """Greedy: longest episode first into the shortest stream. -> dict of (L, E, ...) arrays."""
    streams = [[] for _ in range(n_streams)]
    lens = [0] * n_streams
    for ep in sorted(episodes, key=lambda e: -len(e[1])):
        k = int(np.argmin(lens))
        streams[k].append(ep)
        lens[k] += len(ep[1])
    L, E = max(lens), n_streams
    out = dict(obs=np.zeros((L, E, OBS_SIZE), np.float32), probs=np.zeros((L, E, 6), np.float32),
               rew=np.zeros((L, E), np.float32), done=np.zeros((L, E), bool))
    for e, eps in enumerate(streams):
        t = 0
        for o, a, r in eps:
            n = len(a)
            out["obs"][t:t + n, e] = o
            out["probs"][np.arange(t, t + n), e, a] = 1.0
            out["rew"][t:t + n, e] = r
            out["done"][t + n - 1, e] = True
            t += n
    return out, sum(lens) / (L * E)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, help="results/league_<...>.json (its _replays.jsonl.gz next to it)")
    ap.add_argument("--replays", default=None, help="default: <league>_replays.jsonl.gz")
    ap.add_argument("--top", type=int, default=3, help="best N models of the ranking (kind == model)")
    ap.add_argument("--no-gen", action="store_true", help="skip games on --gen-maps maps")
    ap.add_argument("--streams", type=int, default=0, help="E (default: total steps // 4096, >= 1)")
    ap.add_argument("--out", default="models/checkpoints_D-g2/winners.npz")
    ap.add_argument("--verify", type=int, default=0, help="check K episodes: the model must predict the actions")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    with open(args.league, encoding="utf-8") as f:
        league = json.load(f)
    top = [s["name"] for s in league["ranking"] if s["kind"] == "model"][:args.top]
    spec = dict(zip(league["names"], league["meta"]["specs"]))
    print(f"top-{args.top} models: {top}")
    path = args.replays or os.path.splitext(args.league)[0] + "_replays.jsonl.gz"

    episodes, per_model, checked, models = [], {m: 0 for m in top}, 0, {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rep = json.loads(line)
            meta, w = rep["league"], rep["winner"]
            if w is None or (args.no_gen and meta["map"].startswith("gen:")):
                continue
            name = meta["p0"] if w == 0 else meta["p1"]
            if name not in per_model:
                continue
            ep = episode(rep, w)
            if checked < args.verify:
                assert np.array_equal(predict(models, spec[name], ep[0]), ep[1]), \
                    f"{name}: the model does not reproduce its actions from the rebuilt obs"
                checked += 1
            episodes.append(ep)
            per_model[name] += 1
    if not episodes:
        sys.exit("No won games of the top models in the replays")

    total = sum(len(e[1]) for e in episodes)
    data, fill = pack(episodes, args.streams or max(1, total // 4096))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **data)
    acts = np.concatenate([e[1] for e in episodes])
    print(f"saved {args.out}: {len(episodes)} won episodes, {total} steps, shape {data['obs'].shape[:2]} "
          f"(fill {100 * fill:.1f}%), verified {checked}")
    print("  episodes per model:", per_model)
    print("  action share:", np.round(np.bincount(acts, minlength=6) / len(acts), 3).tolist())


if __name__ == "__main__":
    main()
