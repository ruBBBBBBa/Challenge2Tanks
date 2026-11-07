"""
Deterministic evaluation of a few models against fixed rivals, with series probabilities (PC2).

    python student/eval_vs.py --models models/D2-distill.zip \
        --rivals baseline:hard model:models/shared/A-g7_8000k.zip ... model:models/reserved/R1.zip \
        --out results/eval_D2-distill

Reuses student/league.py: games with league.play (same rows, same random maps: RANDOM_SEED0 + k),
tables with build_table, W-D-L with head_to_head and P(win the best-of-3 series) with series_win_prob.
Maps: the official ones and --random random maps (8, like the league), reported separately.
Model vs model is deterministic, so one game per (map, side); bots get --bot-seeds seeds.
Only plays matches: nothing here goes into a training pool (R1 can be a rival).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from student import league  # noqa: E402
from tanks.maps import list_maps  # noqa: E402

BOTS = {"random", "basic", "aggressive", "defensive"}


def spec_of(x):
    return x if ":" in x or x in BOTS else f"model:{x}"


def short(spec):
    s = spec.split(":", 1)[1] if spec.startswith("model:") else spec
    return os.path.basename(s).replace(".zip", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rivals", nargs="+", required=True)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--bot-seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True, help="output prefix (.md and .json)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)

    specs = [spec_of(m) for m in args.models] + [spec_of(r) for r in args.rivals]
    nm = len(args.models)
    off_keys = [f"official:{m}" for m in list_maps()]
    rnd_keys = [f"random:{league.RANDOM_SEED0 + k}" for k in range(args.random)]
    tasks = []
    for i in range(nm):
        for j in range(nm, len(specs)):
            bot = specs[j].split(":")[0] in BOTS or specs[j].startswith(("my:", "param:"))
            for key in off_keys + rnd_keys:
                for side in (0, 1):
                    for s in range(args.bot_seeds if bot else 1):
                        tasks.append((i, j, key, side, s, specs[i], specs[j]))
    print(f"{nm} models x {len(specs) - nm} rivals x {len(off_keys)}+{len(rnd_keys)} maps x 2 sides: "
          f"{len(tasks)} games", flush=True)
    with Pool(args.procs, initializer=league._worker_init) as pool:
        rows = pool.map(league.play, tasks, chunksize=2)
    table = league.build_table(rows)

    res, lines = {}, []
    sets = [("oficiales", off_keys), (f"aleatorios ({args.random})", rnd_keys)]
    for i in range(nm):
        res[specs[i]] = {}
        lines += [f"## {short(specs[i])}", "",
                  "| rival | " + " | ".join(f"W-D-L {n} | P(serie) {n}" for n, _ in sets) + " |",
                  "|---" * (1 + 2 * len(sets)) + "|"]
        for j in range(nm, len(specs)):
            cells, res[specs[i]][specs[j]] = [], {}
            for n, keys in sets:
                w, d, l = league.head_to_head(table, i, j, keys)
                p = league.series_win_prob(table, i, j, keys)
                res[specs[i]][specs[j]][n] = dict(win=w, draw=d, lose=l, p_series=p)
                cells += [f"{w}-{d}-{l}", f"{100 * p:.0f} %"]
            lines.append(f"| {short(specs[j])} | " + " | ".join(cells) + " |")
        lines.append("")
    md = (f"# Evaluación determinista: {', '.join(short(s) for s in specs[:nm])}\n\n"
          f"Mapas: {len(off_keys)} oficiales y {len(rnd_keys)} aleatorios (semillas {league.RANDOM_SEED0}+k, las de "
          f"la liga), los dos lados. P(serie) = probabilidad de ganar el bo3 con la regla del torneo "
          f"(`league.series_win_prob`).\n\n" + "\n".join(lines))
    print(md)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"specs": specs, "results": res, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
