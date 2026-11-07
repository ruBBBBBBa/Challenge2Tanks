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

--play mode: instead of league replays, play new games between --models (every ordered pair, both
deterministic and stochastic, StudentTanksEnv with --maps / --scenario-prob, random side) and keep the
episodes won by the learner. Each pair is played from both points of view, so every winner is recorded.
Stops at --target kept steps; a model above 50 % of the steps is cut down to the next largest one.
Saves also `episode_start` and a .json with the counts per model, rival and mode. R1 is refused.

    python student/winners_dataset.py --play --target 3000000 --procs 10 --out models/checkpoints_D-g2/winners_play.npz
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter

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


PLAY_MODELS = {   # name -> agent spec (deterministic); "my:stoch:<spec>" is used for the stochastic games
    "A-g9": "model:models/A-g9.zip",
    "A-g10_8000k": "model:models/league/A-g10_8000k.zip",
    "A-g7_8000k": "model:models/candidatos/A-g7_8000k.zip",
    "B-g7_6000k": "model:models/league/B-g7_6000k.zip",
    "C-g7_4500k": "model:models/league/C-g7_4500k.zip",
    "D2b-distill": "model:models/league/D2b-distill.zip",
    "baseline:hard": "baseline:hard",
    "baseline:medium": "baseline:medium",
}


def _no_r1(spec):
    if any(k in spec for k in ("R1", "reserved", "eval_pc4")):
        raise ValueError(f"{spec}: R1 / judges never go into a dataset")


def play_task(task):
    """(learner, lspec, rival, rspec, deterministic, episodes, seed) -> won episodes of the learner + counts."""
    learner, lspec, rival, rspec, det, n_eps, seed, maps, scenario_prob = task
    import torch
    torch.set_num_threads(1)
    from student.envs import StudentTanksEnv
    from tanks.agents.base import load_agent
    env = StudentTanksEnv(opponent=rspec if det else f"my:stoch:{rspec}", maps=maps, scenario_prob=scenario_prob)
    agent = load_agent(lspec, deterministic=det)
    env.reset(seed=seed)
    won, results = [], Counter()
    for _ in range(n_eps):
        obs, _ = env.reset()
        agent.reset()
        O, A, R, done = [], [], [], False
        while not done:
            a = agent.act(env._state)          # the same vector_obs the env returns
            O.append(obs)
            A.append(a)
            obs, r, term, trunc, info = env.step(a)
            R.append(r)
            done = term or trunc
        results[info["result"]] += 1
        if info["result"] == "win":
            won.append((np.array(O, np.float32), np.array(A, np.int64), np.array(R, np.float32)))
    return learner, rival, det, won, dict(results)


def play_dataset(args):
    from multiprocessing import Pool
    models = {k: v for k, v in PLAY_MODELS.items() if not args.models or k in args.models}
    for spec in models.values():
        _no_r1(spec)
    combos = [(l, r, det) for l in models for r in models if l != r for det in (True, False)]
    eps, games = [], Counter()     # eps: (learner, rival, det, obs, acts, rews)
    rnd, seed = np.random.default_rng(args.seed), args.seed
    t0 = time.time()

    def balanced():
        """Cut the dominant learner (> 50 % of the steps) down to the next one, until nobody dominates."""
        kept, cut = list(eps), []
        for _ in range(10):
            steps = Counter()
            for e in kept:
                steps[e[0]] += len(e[4])
            if len(steps) < 2:
                break
            (top, n), (_, cap) = steps.most_common(2)
            if n <= 0.5 * sum(steps.values()):
                break
            idx = [k for k, e in enumerate(kept) if e[0] == top]
            keep, total = set(), 0
            for k in rnd.permutation(idx):
                if total + len(kept[k][4]) <= cap:
                    keep.add(k)
                    total += len(kept[k][4])
            kept = [e for k, e in enumerate(kept) if e[0] != top or k in keep]
            cut.append(top)
        return kept, ",".join(dict.fromkeys(cut))

    with Pool(args.procs) as pool:
        while True:
            tasks = []
            for (l, r, det) in combos:
                seed += 1
                tasks.append((l, models[l], r, models[r], det, args.episodes_per_task, seed, args.maps,
                              args.scenario_prob))
            for l, r, det, won, res in pool.imap_unordered(play_task, tasks):
                games[(l, r, det)] += sum(res.values())
                games[(l, r, det, "win")] += res.get("win", 0)
                eps.extend((l, r, det, o, a, rw) for o, a, rw in won)
            kept, cut = balanced()
            total = sum(len(e[4]) for e in kept)
            print(f"[play] {sum(len(e[4]) for e in eps)} won steps, {total} after balancing"
                  f"{' (cut ' + cut + ')' if cut else ''} · {(time.time() - t0) / 60:.1f} min", flush=True)
            if total >= args.target:
                eps = kept
                break

    data, fill = pack([(o, a, rw) for _, _, _, o, a, rw in eps], args.streams or 64)
    data["episode_start"] = np.zeros_like(data["done"])
    data["episode_start"][0] = True
    data["episode_start"][1:] = data["done"][:-1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    steps = {name: Counter() for name in ("learner", "rival", "mode")}
    for e in eps:
        steps["learner"][e[0]] += len(e[4])
        steps["rival"][e[1]] += len(e[4])
        steps["mode"]["det" if e[2] else "stoch"] += len(e[4])
    meta = dict(models=models, maps=args.maps, scenario_prob=args.scenario_prob, target=args.target,
                steps=sum(len(e[4]) for e in eps), episodes=len(eps), fill=fill, shape=list(data["obs"].shape),
                steps_by_learner=dict(steps["learner"]), steps_by_rival=dict(steps["rival"]),
                steps_by_mode=dict(steps["mode"]),
                win_rate={f"{l} vs {r} ({'det' if det else 'stoch'})": games[(l, r, det, "win")] / max(1, games[(l, r, det)])
                          for l, r, det in combos},
                minutes=(time.time() - t0) / 60)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 2**20:.0f} MB): {meta['steps']} steps, {len(eps)} episodes")
    for k in ("learner", "rival", "mode"):
        tot = sum(steps[k].values())
        print(f"  by {k}: " + ", ".join(f"{n} {v} ({v / tot:.0%})" for n, v in steps[k].most_common()))


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
    ap.add_argument("--league", help="results/league_<...>.json (its _replays.jsonl.gz next to it)")
    ap.add_argument("--play", action="store_true", help="play new games instead of reading league replays")
    ap.add_argument("--models", nargs="*", default=None, help="--play: subset of PLAY_MODELS")
    ap.add_argument("--maps", default="mixed2")
    ap.add_argument("--scenario-prob", type=float, default=0.3)
    ap.add_argument("--target", type=int, default=3_000_000, help="--play: kept steps")
    ap.add_argument("--episodes-per-task", type=int, default=4)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replays", default=None, help="default: <league>_replays.jsonl.gz")
    ap.add_argument("--top", type=int, default=3, help="best N models of the ranking (kind == model)")
    ap.add_argument("--no-gen", action="store_true", help="skip games on --gen-maps maps")
    ap.add_argument("--streams", type=int, default=0, help="E (default: total steps // 4096, >= 1)")
    ap.add_argument("--out", default="models/checkpoints_D-g2/winners.npz")
    ap.add_argument("--verify", type=int, default=0, help="check K episodes: the model must predict the actions")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.play:
        return play_dataset(args)
    if not args.league:
        ap.error("--league is required (or use --play)")

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
