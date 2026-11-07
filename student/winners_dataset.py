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

--play-replays mode (dataset for D3, what goes to git): stochastic games between PLAY_MODELS on the maps of a
league (--maps-from <league.json>), both sides, from the normal start (so a replay rebuilds them exactly).
Every player computes, at each step and with the same input state (LSTM state included), the deterministic
action (the label) and the sampled one (the one played). Only decisive games are kept: the replay plus the
winner's deterministic labels. No model above --max-share of the winner steps. Output: one .jsonl.gz.
--from-play rebuilds the distill.py .npz from it (obs from the engine, probs = one-hot of the labels).

    python student/winners_dataset.py --play-replays --maps-from results/league_..._d3data.json --target 2500000 \
        --out results/d3play_replays.jsonl.gz
    python student/winners_dataset.py --from-play results/d3play_replays.jsonl.gz --verify 20 --out <dataset>.npz
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


def spec_path(spec):
    from tanks.agents.base import BASELINES_DIR
    if spec.startswith("baseline:"):
        return os.path.join(BASELINES_DIR, spec.split(":", 1)[1] + ".zip")
    return spec.split(":", 1)[1] if spec.startswith("model:") else spec


def labeling_agent(spec, name):
    """ModelAgent that plays the sampled action and records the deterministic one (same input state)."""
    from tanks.agents.model_agent import ModelAgent

    class LabelingAgent(ModelAgent):
        def reset(self, seed=None):
            super().reset(seed)
            self.labels = []

        def act(self, state):
            self._ensure_loaded()
            obs = vector_obs(state)
            det, new_state = self.model.predict(obs, state=self._rnn_state, episode_start=self._episode_start,
                                                deterministic=True)
            smp, _ = self.model.predict(obs, state=self._rnn_state, episode_start=self._episode_start,
                                        deterministic=False)
            self._rnn_state = new_state          # the LSTM state does not depend on how the action is picked
            self._episode_start = np.zeros((1,), dtype=bool)
            self.labels.append(int(np.asarray(det).item()))
            return int(np.asarray(smp).item())

    return LabelingAgent(spec_path(spec), deterministic=False, name=name)


_LAGENTS = {}


def play_replay_task(task):
    """(name0, spec0, name1, spec1, map_key, seed) -> replay line (decisive games only) or None, + steps."""
    n0, s0, n1, s1, key, seed = task
    import torch
    torch.set_num_threads(1)
    from student.league import resolve_map
    from tanks.match import REPLAY_VERSION, play_match
    agents = []
    for n, s in ((n0, s0), (n1, s1)):
        if n not in _LAGENTS:
            _LAGENTS[n] = labeling_agent(s, n)
        agents.append(_LAGENTS[n])
    res = play_match(agents[0], agents[1], resolve_map(key), seed=seed)
    if res["winner"] is None:
        return None, res["steps"]
    w = res["winner"]
    labels = agents[w].labels
    assert len(labels) == len(res["actions"])
    rep = {k: v for k, v in res.items() if k != "frames"}
    rep.update(version=REPLAY_VERSION, league=dict(p0=n0, p1=n1, map=key, seed=seed, winner_name=(n0, n1)[w]),
               labels="".join(map(str, labels)))
    return rep, res["steps"]


def play_replays(args):
    import gzip as gz
    from multiprocessing import Pool
    models = {k: v for k, v in PLAY_MODELS.items() if not args.models or k in args.models}
    for spec in models.values():
        _no_r1(spec)
    with open(args.maps_from, encoding="utf-8") as f:
        maps = json.load(f)["meta"]["maps"]
    names = list(models)
    pairs = [(a, b) for k, a in enumerate(names) for b in names[k + 1:]]
    reps, steps, played, seed, t0 = [], Counter(), Counter(), args.seed, time.time()
    rng = np.random.default_rng(args.seed)

    def balanced(lines):
        """Cap every winner at C with C = max_share * sum(min(steps_i, C)) (water filling), so that nobody
        ends above max_share. If that is impossible (too few winners), the dominant ones are cut to the next."""
        st = Counter()
        for r in lines:
            st[r["league"]["winner_name"]] += len(r["labels"])
        if len(st) < 2:
            return list(lines), st
        cap = args.max_share * sum(st.values())
        for _ in range(100):
            new = args.max_share * sum(min(v, cap) for v in st.values())
            if abs(new - cap) < 1:
                break
            cap = new
        if cap < 0.5 * sorted(st.values())[-2]:          # infeasible share: equalise with the runner-up
            cap = sorted(st.values())[-2]
        kept, acc = [], Counter()
        for k in rng.permutation(len(lines)):
            r = lines[k]
            m = r["league"]["winner_name"]
            if acc[m] + len(r["labels"]) <= cap:
                kept.append(r)
                acc[m] += len(r["labels"])
        return kept, acc

    with Pool(args.procs) as pool:
        while True:
            tasks = []
            for a, b in pairs:
                for key in maps:
                    for side in (0, 1):
                        seed += 1
                        n0, n1 = (a, b) if side == 0 else (b, a)
                        tasks.append((n0, models[n0], n1, models[n1], key, seed))
            for rep, n in pool.imap_unordered(play_replay_task, tasks, chunksize=8):
                played["games"] += 1
                played["steps"] += n
                if rep is not None:
                    reps.append(rep)
            kept, st = balanced(reps)
            total = sum(st.values())
            print(f"[play-replays] {played['games']} games, {len(reps)} decisive, {total} winner steps kept "
                  f"· {(time.time() - t0) / 60:.1f} min", flush=True)
            if total >= args.target:
                break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gz.open(args.out, "wt", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    by_rival, games_won = Counter(), Counter()
    for r in kept:
        m = r["league"]
        by_rival[m["p1"] if m["winner_name"] == m["p0"] else m["p0"]] += len(r["labels"])
        games_won[m["winner_name"]] += 1
    agree = sum(sum(a == b for a, b in zip(r["labels"], (str(x[r["winner"]]) for x in r["actions"]))) for r in kept)
    meta = dict(models=models, maps=maps, max_share=args.max_share, games=played["games"], decisive=len(reps),
                kept_games=len(kept), winner_steps=total, steps_by_winner=dict(st), games_won=dict(games_won),
                steps_by_loser=dict(by_rival), label_equals_played=agree / max(1, total),
                minutes=(time.time() - t0) / 60, size_mb=os.path.getsize(args.out) / 2**20)
    with open(args.out.replace(".jsonl.gz", "") + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.out} ({meta['size_mb']:.1f} MB): {len(kept)} games, {total} winner steps; "
          f"label == played action in {meta['label_equals_played']:.0%} of the steps")
    for k, c in (("winner", st), ("loser", by_rival)):
        print(f"  steps by {k}: " + ", ".join(f"{n} {v} ({v / total:.0%})" for n, v in c.most_common()))


def from_play(args):
    """Rebuild the distill.py dataset from a --play-replays file: obs from the engine, labels as probs."""
    import gzip as gz
    episodes, checked, cache = [], 0, {}
    models = {}
    meta_path = args.from_play.replace(".jsonl.gz", "") + ".json"
    if os.path.exists(meta_path):
        models = json.load(open(meta_path, encoding="utf-8"))["models"]
    for line in gz.open(args.from_play, "rt", encoding="utf-8"):
        rep = json.loads(line)
        obs, _, rews = episode(rep, rep["winner"])
        labels = np.frombuffer(rep["labels"].encode(), dtype=np.uint8) - ord("0")
        if checked < args.verify and models:
            pred = predict(cache, spec_path(models[rep["league"]["winner_name"]]), obs)   # LSTM state carried
            assert np.array_equal(pred, labels), f"labels do not match the deterministic {rep['league']['winner_name']}"
            checked += 1
        episodes.append((obs, labels.astype(np.int64), rews))
    total = sum(len(e[1]) for e in episodes)
    data, fill = pack(episodes, args.streams or max(1, total // 4096))
    data["episode_start"] = np.zeros_like(data["done"])
    data["episode_start"][0] = True
    data["episode_start"][1:] = data["done"][:-1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **data)
    print(f"saved {args.out}: {len(episodes)} episodes, {total} steps, shape {data['obs'].shape[:2]} "
          f"(fill {100 * fill:.1f}%), verified {checked} episodes against the deterministic winner")


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
    ap.add_argument("--play-replays", action="store_true", help="stochastic games -> replays + deterministic labels")
    ap.add_argument("--maps-from", default=None, help="--play-replays: league .json whose maps are used")
    ap.add_argument("--max-share", type=float, default=0.35, help="--play-replays: max share of one winner")
    ap.add_argument("--from-play", default=None, help="rebuild the .npz from a --play-replays .jsonl.gz")
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
    if args.play_replays:
        return play_replays(args)
    if args.from_play:
        return from_play(args)
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
