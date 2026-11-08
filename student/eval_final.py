"""
Final evaluation with the league.py functions: round robin among the candidates (deterministic, both sides,
one game per (map, side)), plus every candidate against the judge (R1, read in place, never in a pool).
Map sets reported apart: official / random (fixed seeds) / generated (map_gen.py).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python student/eval_final.py --out results/eval_final.md
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import (RANDOM_SEED0, ROOT, _worker_init, bradley_terry, build_table, head_to_head,  # noqa: E402
                    list_maps, pct, play, series_win_prob)

MODELS = {"A-g17_8000k": "models/league/A-g17_8000k.zip", "A-g18_8000k": "models/league/A-g18_8000k.zip",
          "A-g19": "models/A-g19.zip", "D2b-distill": "models/league/D2b-distill.zip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=24)
    ap.add_argument("--gen-maps", type=int, default=16)
    ap.add_argument("--gen-seed", type=int, default=20261108)
    ap.add_argument("--judge", default="models/eval_pc4/R1.zip")
    ap.add_argument("--reference", default="baseline:hard", help="extra participant ('' for none)")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default="results/eval_final.md")
    args = ap.parse_args()
    os.chdir(ROOT)
    assert "/R" not in "".join(MODELS.values()), "a judge cannot be a participant"
    names = list(MODELS) + ([args.reference] if args.reference else [])
    specs = [f"model:{p}" for p in MODELS.values()] + ([args.reference] if args.reference else [])
    judge = os.path.splitext(os.path.basename(args.judge))[0]
    names.append(judge), specs.append(f"model:{args.judge}")
    J = len(names) - 1
    sets = {"oficiales": [f"official:{m}" for m in list_maps()],
            "aleatorios": [f"random:{RANDOM_SEED0 + k}" for k in range(args.random)],
            "generados": [f"gen:{args.gen_seed}:{k}" for k in range(args.gen_maps)]}
    keys = [k for v in sets.values() for k in v]
    tasks = [(i, j, m, s, 0, specs[i], specs[j]) for i, j in itertools.combinations(range(len(names)), 2)
             for m in keys for s in (0, 1)]
    print(f"[eval_final] {len(names)} participants (judge {judge}), {len(keys)} maps, {len(tasks)} games", flush=True)
    t0 = time.time()
    with Pool(args.procs, initializer=_worker_init, initargs=(False,)) as pool:
        rows = list(pool.imap_unordered(play, tasks, chunksize=2))
    table = build_table(rows)
    cand = list(range(J))                                   # the judge does not enter the Elo
    wdl = lambda w: f"{w[0]}-{w[1]}-{w[2]}"                 # noqa: E731
    score = lambda w: (w[0] + 0.5 * w[1]) / max(1, sum(w))  # noqa: E731

    L = [f"# Evaluación final ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"Todos contra todos determinista con las funciones de `student/league.py` (`play`, `build_table`, "
         f"`head_to_head`, `series_win_prob`, `bradley_terry`): {len(sets['oficiales'])} mapas oficiales, "
         f"{args.random} aleatorios (semillas {RANDOM_SEED0}+) y {args.gen_maps} generados (`map_gen.py`, semilla "
         f"{args.gen_seed}), los dos lados y una partida por (mapa, lado). {len(rows)} partidas en "
         f"{(time.time() - t0) / 60:.1f} min. `{args.reference}` va como referencia. **{judge}** "
         f"(`{args.judge}`) es solo juez: se lee en su sitio, no entra en el Elo ni en ningún pool.", "",
         "P(serie) = probabilidad de ganar un mejor de 3 con la regla del README (los empates se repiten en la "
         "siguiente combinación mapa/lado, luego daño, luego moneda), 2000 series simuladas.", "",
         "## Ranking (Elo Bradley–Terry por conjunto de mapas; hard = referencia)", "",
         "| modelo | Elo total | Elo oficiales | Elo aleatorios | Elo generados | W-D-L total | puntuación | "
         "empates | daño medio | timeouts sin daño |", "|---|---|---|---|---|---|---|---|---|---|"]
    elo = {}
    for name, ks in [("total", keys)] + list(sets.items()):
        ks = set(ks)
        games = [(r["i"], r["j"], {"win": 1, "lose": 0, None: 0.5}[r["result"]]) for r in rows
                 if r["map"] in ks and r["i"] in cand and r["j"] in cand]
        e = bradley_terry(J, games)
        elo[name] = e - (e[names.index(args.reference)] - 1500 if args.reference else 0)
    order = sorted(cand, key=lambda i: -elo["total"][i])
    for i in order:
        mine = [r for r in rows if J not in (r["i"], r["j"]) and i in (r["i"], r["j"])]
        w = [sum(head_to_head(table, i, j, keys)[k] for j in cand if j != i) for k in range(3)]
        dmg = np.mean([r["dmg"] if r["i"] == i else r["dmg_en"] for r in mine])
        zero = sum(r["end"] == "timeout" and r["dmg"] == 0 and r["dmg_en"] == 0 for r in mine)
        L.append(f"| {names[i]} | {elo['total'][i]:.0f} | {elo['oficiales'][i]:.0f} | {elo['aleatorios'][i]:.0f} | "
                 f"{elo['generados'][i]:.0f} | {wdl(w)} | {pct(score(w))} | {pct(w[1] / max(1, sum(w)))} | "
                 f"{dmg:.2f} | {zero} |")

    for sname, ks in sets.items():
        L += ["", f"## Cara a cara en mapas {sname} (fila contra columna: W-D-L · P(serie))", "",
              "| | " + " | ".join(names[j] for j in order) + " |", "|---" * (len(order) + 1) + "|"]
        for i in order:
            cells = ["·" if i == j else f"{wdl(head_to_head(table, i, j, ks))} · {pct(series_win_prob(table, i, j, ks))}"
                     for j in order]
            L.append(f"| {names[i]} | " + " | ".join(cells) + " |")

    L += ["", f"## Juez {judge}", "", "| modelo | " + " | ".join(f"W-D-L {s} | P(serie) {s}" for s in sets) + " |",
          "|---" * (1 + 2 * len(sets)) + "|"]
    for i in order:
        L.append(f"| {names[i]} | " + " | ".join(f"{wdl(head_to_head(table, i, J, ks))} | "
                                                f"{pct(series_win_prob(table, i, J, ks))}" for ks in sets.values()) + " |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(dict(names=names, specs=specs, judge=judge, sets=sets, games=rows,
                       elo={k: list(map(float, v)) for k, v in elo.items()}), f)
    print("\n".join(L))


if __name__ == "__main__":
    main()
