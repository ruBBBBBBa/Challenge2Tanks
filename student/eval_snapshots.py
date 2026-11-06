"""
Deterministic evaluation of snapshots against fixed rivals (PC2, rama E).

    python student/eval_snapshots.py --models models/league/E-g1_*k.zip models/shared/A-g4_4000k.zip \
        --rivals model:models/shared/A-g4_4000k.zip model:models/shared/B-g1_20000k.zip \
                 model:models/shared/C-g1_20000k.zip baseline:hard \
        --ref models/shared/A-g4_4000k.zip --out results/E-g1_eval --outbox E

40 games per (model, rival): 16 official_aug maps + 4 random maps (fixed seeds) x 2 sides.
Every match is deterministic (both agents deterministic), so one game per (map, side) suffices.
Best = fewest draws among the models that lose no more than --ref (total over all rivals).
"""
import argparse
import glob
import json
import os
import shutil
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N_RANDOM = 4
RANDOM_SEED0 = 20261106


def _maps():
    from student.envs import resolve_maps
    from tanks.game import GameMap
    from tanks.maps.generator import random_map
    import numpy as np
    maps = [(os.path.basename(p)[:-4], p) for p in resolve_maps("official_aug")]
    for k in range(N_RANDOM):
        maps.append((f"random_{k}", random_map(np.random.default_rng(RANDOM_SEED0 + k))))
    return maps


_AGENTS = {}


def _agent(spec):
    from tanks.agents.base import load_agent
    if spec not in _AGENTS:
        _AGENTS[spec] = load_agent(spec)
    return _AGENTS[spec]


def _init():
    import torch
    torch.set_num_threads(1)


def _play(job):
    from tanks.match import play_match
    model, rival, mi, side = job
    name, gmap = _maps_cache()[mi]
    a, b = _agent(f"model:{model}"), _agent(rival)
    players = (a, b) if side == 0 else (b, a)
    r = play_match(*players, game_map=gmap, seed=mi)
    res = "draw" if r["winner"] is None else ("win" if r["winner"] == side else "lose")
    return model, rival, name, side, res, int(r["damage"][side]), int(r["steps"])


_MAPS = None


def _maps_cache():
    global _MAPS
    if _MAPS is None:
        _MAPS = _maps()
    return _MAPS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rivals", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="reference model: candidates may not lose more than it")
    ap.add_argument("--out", required=True, help="output prefix (.md and .json)")
    ap.add_argument("--outbox", default=None, help="if set, copy the best (not the ref) to models/outbox/")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")    # Windows console is cp1252
    os.chdir(ROOT)
    models = []
    for m in args.models:
        for p in sorted(glob.glob(m)) or [m]:
            p = p.replace(os.sep, "/")
            if p not in models:
                models.append(p)
    ref = args.ref.replace(os.sep, "/")
    if ref not in models:
        models.append(ref)
    n_maps = len(_maps_cache())
    jobs = [(m, r, mi, s) for m in models for r in args.rivals for mi in range(n_maps) for s in (0, 1)]
    print(f"{len(models)} models x {len(args.rivals)} rivals x {n_maps} maps x 2 sides = {len(jobs)} games", flush=True)
    with Pool(args.procs, initializer=_init) as pool:
        out = pool.map(_play, jobs, chunksize=4)

    table = {m: {r: {"win": 0, "draw": 0, "lose": 0, "dmg": 0} for r in args.rivals} for m in models}
    for m, r, _, _, res, dmg, _ in out:
        table[m][r][res] += 1
        table[m][r]["dmg"] += dmg
    tot = {m: {k: sum(table[m][r][k] for r in args.rivals) for k in ("win", "draw", "lose")} for m in models}
    ok = [m for m in models if m != ref and tot[m]["lose"] <= tot[ref]["lose"]]
    best = min(ok, key=lambda m: (tot[m]["draw"], tot[m]["lose"], -tot[m]["win"])) if ok else None

    games = 2 * n_maps
    lines = [f"# Evaluación determinista ({games} partidas por rival: 16 official_aug + {N_RANDOM} aleatorios × 2 lados)",
             "", "W-D-L por rival. Referencia: `" + ref + "`.", "",
             "| modelo | " + " | ".join(r.replace("model:models/shared/", "").replace(".zip", "") for r in args.rivals)
             + " | total W-D-L |", "|---" * (len(args.rivals) + 2) + "|"]
    for m in models:
        cells = [f"{table[m][r]['win']}-{table[m][r]['draw']}-{table[m][r]['lose']}" for r in args.rivals]
        mark = " (ref)" if m == ref else (" **← mejor**" if m == best else "")
        lines.append(f"| `{os.path.basename(m)}`{mark} | " + " | ".join(cells)
                     + f" | {tot[m]['win']}-{tot[m]['draw']}-{tot[m]['lose']} |")
    lines += ["", f"Mejor (menos empates sin perder más que la referencia, {tot[ref]['lose']} derrotas): "
              + (f"`{best}`" if best else "ninguno")]
    md = "\n".join(lines)
    print(md)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"table": table, "total": tot, "ref": ref, "best": best, "games_per_rival": games,
                   "games": [dict(zip(("model", "rival", "map", "side", "result", "damage", "steps"), g))
                             for g in out]}, f, indent=1)
    if best and args.outbox:
        os.makedirs("models/outbox", exist_ok=True)
        dst = os.path.join("models/outbox", os.path.basename(best))
        shutil.copy2(best, dst)
        print(f"copied {best} -> {dst}")


if __name__ == "__main__":
    main()
