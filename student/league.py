"""
League: deterministic round robin between models, baselines and bots (CLAUDE.md section 6).

    python student/league.py --models "models/league/A-g1_*k.zip" models/B-g1.zip \
        --baselines hard medium --bots random basic aggressive defensive my:rusher my:bushcamper my:hitandhide \
        --procs 8 --replays-dir replays/league

Model vs model is deterministic: one game per (map, side). Any pair with a bot plays --bot-seeds games.
Maps: official, official_aug variants (student/maps) and --random N fixed-seed random maps (reported apart).
--gen-maps N adds N maps from student/map_gen.py (seed --gen-seed, not the one of student/maps/gen):
they get their own column (win rate and a separate Elo) and are left out of the main Elo and stats.
Output: results/league_<date>.md and .json (Bradley-Terry Elo, win rate vs top-3, worst matchup,
random maps, draws, damage, suspicious games, simulated bracket, behaviour stats).
Behaviour stats come from re-simulating every game from its actions (the engine is deterministic).
--save-all-replays writes every game (replay format + league metadata) to <output>_replays.jsonl.gz,
the input of student/winners_dataset.py.
"""
import argparse
import datetime
import glob
import gzip
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from student.envs import _official_aug  # noqa: E402
from student.map_gen import STYLES, gen_map  # noqa: E402
from tanks.agents.base import load_agent  # noqa: E402
from tanks.game import BUSH, GameConfig, TanksGame  # noqa: E402
from tanks.maps import list_maps, load_map, random_map  # noqa: E402
from tanks.match import REPLAY_VERSION, play_match, save_replay  # noqa: E402

RANDOM_SEED0 = 10_000   # random map i uses np.random.default_rng(RANDOM_SEED0 + i)


# ---------------------------------------------------------------- games (workers)
def resolve_map(key):
    kind, value = key.split(":", 1)
    if kind == "random":
        return random_map(np.random.default_rng(int(value)))
    if kind == "gen":   # "gen:<seed>:<k>", one independent RNG per map
        seed, k = map(int, value.split(":"))
        return gen_map(np.random.default_rng([seed, k]), STYLES[k % len(STYLES)], name=f"gen_{seed}_{k}")
    return load_map(value)


_AGENTS = {}


def _agent(spec):
    if spec not in _AGENTS:
        _AGENTS[spec] = load_agent(spec)
    return _AGENTS[spec]


_SAVE_REPLAYS = False


def _worker_init(save_replays=False):
    global _SAVE_REPLAYS
    import torch
    torch.set_num_threads(1)
    _SAVE_REPLAYS = save_replays


def replay_events(game_map, config, actions):
    """Re-simulate a game from its actions. Yields (game, None) first, then (game, events) after each step."""
    game = TanksGame(game_map, GameConfig.from_dict(config))
    yield game, None
    for acts in actions:
        events = game.step(acts)
        yield game, events


def behaviour(game_map, res):
    """Per player: shots, cancelled bullets, steps hidden in a bush (on a bush and invisible to the enemy)."""
    shots, canc, hidden = [0, 0], [0, 0], [0, 0]
    for game, ev in replay_events(game_map, res["config"], res["actions"]):
        if ev is None:
            continue
        for p in (0, 1):
            shots[p] += ev[p]["shots_fired"]
            canc[p] += ev[p]["bullets_cancelled"]
            t = game.tanks[p]
            hidden[p] += int(game.tiles[t.y, t.x] == BUSH and not game.enemy_visible(1 - p))
    assert game.winner == res["winner"] and shots == res["shots"], "replay diverged from the game"
    return shots, canc, hidden


def play(task, full=False):
    """task = (i, j, map_key, side, seed): participant i plays side `side` against j."""
    i, j, key, side, seed, si, sj = task
    a, b = (_agent(si), _agent(sj)) if side == 0 else (_agent(sj), _agent(si))
    game_map = resolve_map(key)
    res = play_match(a, b, game_map, seed=seed)
    if full:
        return res
    me, en = side, 1 - side
    shots, canc, hidden = behaviour(game_map, res)
    row = dict(i=i, j=j, map=key, side=side, seed=seed,
               result=None if res["winner"] is None else ("win" if res["winner"] == me else "lose"),
               dmg=res["damage"][me], dmg_en=res["damage"][en], steps=res["steps"], end=res["end_reason"],
               shots=shots[me], shots_en=shots[en], canc=canc[me], canc_en=canc[en],
               hidden=hidden[me], hidden_en=hidden[en])
    if _SAVE_REPLAYS:
        row["replay"] = dict({k: v for k, v in res.items() if k != "frames"}, version=REPLAY_VERSION)
    return row


# --------------------------------------------------------------------- analysis
def bradley_terry(n, games, iters=500):
    """Elo from (i, j, score_i) with score in {1, 0.5, 0}. One virtual draw per pair as prior."""
    w = np.zeros((n, n))
    for i, j, s in games:
        w[i, j] += s
        w[j, i] += 1 - s
    played = (w + w.T) > 0
    w += 0.5 * played
    nij = w + w.T
    p = np.ones(n)
    for _ in range(iters):
        denom = (nij / (p[:, None] + p[None, :])).sum(1)
        p = np.where(denom > 0, w.sum(1) / np.maximum(denom, 1e-12), p)
        p /= np.exp(np.log(p).mean())
    return 400 * np.log10(p)


def bo3(rng, a, b, maps, table):
    """Best of 3 as in the README: draws do not count, tie-break by damage dealt.
    Drawn games are replayed on the next (map, side) combination of the set (sides alternate) until someone
    wins 2 or every distinct combination has been played (2 x len(maps): after that the deterministic games
    would repeat). Then: more wins; else more total damage over the series; else a coin.
    Returns (winner, how) with how in {"wins", "damage", "coin"}."""
    order = [maps[k] for k in rng.permutation(len(maps))]
    s0 = int(rng.integers(2))
    # two passes over the maps with sides alternating game after game; with an even number of maps the
    # second pass is shifted by one map so that every (map, side) appears exactly once
    second = order[1:] + order[:1] if len(order) % 2 == 0 else order
    combos = [(m, (s0 + g) % 2) for g, m in enumerate(order + second)]
    wins, dmg = [0, 0], [0, 0]
    for m, side in combos:
        r = table[(a, b, m, side)]
        r = r[int(rng.integers(len(r)))]
        if r["result"] == "win":
            wins[0] += 1
        elif r["result"] == "lose":
            wins[1] += 1
        dmg[0] += r["dmg"]
        dmg[1] += r["dmg_en"]
        if max(wins) == 2:
            break
    if wins[0] != wins[1]:
        return (a if wins[0] > wins[1] else b), "wins"
    if dmg[0] != dmg[1]:
        return (a if dmg[0] > dmg[1] else b), "damage"
    return (a if rng.random() < 0.5 else b), "coin"


def simulate_bracket(entrants, table, official_maps, final_maps, sims, seed=0):
    """-> ({p: P(champion)}, {p: share of p's series decided by a coin}, overall share of coin series)."""
    rng = np.random.default_rng(seed)
    champs, series, coins = defaultdict(int), defaultdict(int), defaultdict(int)
    for _ in range(sims):
        alive = list(rng.permutation(entrants))
        while len(alive) > 1:
            maps = final_maps if len(alive) == 2 else official_maps
            nxt = []
            for k in range(0, len(alive), 2):
                w, how = bo3(rng, alive[k], alive[k + 1], maps, table)
                nxt.append(w)
                for p in (alive[k], alive[k + 1]):
                    series[int(p)] += 1
                    coins[int(p)] += how == "coin"
            alive = nxt
        champs[int(alive[0])] += 1
    overall = sum(coins.values()) / max(1, sum(series.values()))
    return ({i: champs[i] / sims for i in entrants}, {i: coins[i] / max(1, series[i]) for i in entrants}, overall)


OUTSIDE_SERIES = 2000   # simulated series per (A model, outside rival, map set)
FLIP = {"win": "lose", "lose": "win", None: None}


def build_table(rows):
    """{(a, b, map, side of a): [rows from a's view]} from league rows (both directions)."""
    table = defaultdict(list)
    for r in rows:
        table[(r["i"], r["j"], r["map"], r["side"])].append(r)
        table[(r["j"], r["i"], r["map"], 1 - r["side"])].append(
            dict(r, result=FLIP[r["result"]], dmg=r["dmg_en"], dmg_en=r["dmg"]))
    return table


def series_win_prob(table, a, b, maps, n=OUTSIDE_SERIES, seed=1):
    """P(a wins a best-of-3 series against b on `maps`) with the README rule (see bo3), by simulation."""
    rng = np.random.default_rng(seed)
    return float(np.mean([bo3(rng, a, b, maps, table)[0] == a for _ in range(n)]))


def head_to_head(table, a, b, maps):
    """(wins, draws, losses) of a against b over every (map, side) of `maps`."""
    res = [r["result"] for m in maps for side in (0, 1) for r in table.get((a, b, m, side), [])]
    return res.count("win"), res.count(None), res.count("lose")



def analyse(names, kinds, all_rows, official_maps, random_maps, sims):
    n = len(names)
    score = {"win": 1.0, "draw": 0.5, "lose": 0.0}
    rows = [r for r in all_rows if not r["map"].startswith("gen:")]
    gen_rows = [r for r in all_rows if r["map"].startswith("gen:")]
    elo = bradley_terry(n, [(r["i"], r["j"], score[r["result"] or "draw"]) for r in rows])
    elo_gen = bradley_terry(n, [(r["i"], r["j"], score[r["result"] or "draw"]) for r in gen_rows])
    if "baseline:hard" in names:
        elo = elo - elo[names.index("baseline:hard")] + 1500
    else:
        elo = elo - elo.mean() + 1500
    elo_gen = elo_gen - elo_gen.mean() + elo.mean() if gen_rows else None
    order = list(np.argsort(-elo))

    # results seen from both sides: per[p] = list of (opponent, result, dmg, map)
    flip = FLIP
    per = defaultdict(list)
    for r in rows:
        per[r["i"]].append((r["j"], r["result"], r["dmg"], r["map"]))
        per[r["j"]].append((r["i"], flip[r["result"]], r["dmg_en"], r["map"]))
    table = build_table(rows)

    def wr(lst):
        return sum(x[1] == "win" for x in lst) / len(lst) if lst else float("nan")

    stats = []
    for p in range(n):
        lst = per[p]
        top3 = [q for q in order if q != p][:3]
        by_opp = defaultdict(list)
        for x in lst:
            by_opp[x[0]].append(score[x[1] or "draw"])
        worst = min(by_opp, key=lambda q: np.mean(by_opp[q])) if by_opp else None
        stats.append(dict(
            name=names[p], kind=kinds[p], elo=float(elo[p]), games=len(lst),
            win=sum(x[1] == "win" for x in lst), draw=sum(x[1] is None for x in lst),
            lose=sum(x[1] == "lose" for x in lst), win_rate=wr(lst),
            wr_top3=wr([x for x in lst if x[0] in top3]),
            worst=None if worst is None else names[worst],
            worst_score=None if worst is None else float(np.mean(by_opp[worst])),
            wr_random=wr([x for x in lst if x[3].startswith("random:")]),
            wr_official=wr([x for x in lst if x[3] in official_maps]),
            avg_damage=float(np.mean([x[2] for x in lst])) if lst else 0.0,
            wr_gen=wr([(None, r["result"] if r["i"] == p else flip[r["result"]]) for r in gen_rows
                       if p in (r["i"], r["j"])]),
            elo_gen=None if elo_gen is None else float(elo_gen[p])))

    suspicious = [r for r in all_rows if r["end"] == "timeout" and r["dmg"] + r["dmg_en"] == 0]
    for r in suspicious:
        for p in (r["i"], r["j"]):
            stats[p]["suspicious"] = stats[p].get("suspicious", 0) + 1

    # bracket: the 8 best non-bots (largest power of two available); final on random maps
    contenders = [p for p in order if kinds[p] != "bot"]
    size = 2 ** int(np.log2(len(contenders))) if len(contenders) >= 2 else 0
    size = min(size, 8)
    entrants = contenders[:size]
    final_maps = random_maps or official_maps
    champ, coin, coin_all = simulate_bracket(entrants, table, official_maps or final_maps, final_maps, sims) \
        if size >= 2 else ({}, {}, None)
    for p, c in champ.items():
        stats[p]["p_champion"] = c
        stats[p]["coin_series"] = coin[p]
    for s in stats:
        s["coin_series_all"] = coin_all

    matrix = {names[p]: {names[q]: float(np.mean([score[x[1] or "draw"] for x in per[p] if x[0] == q]))
                         for q in order[:10] if q != p and any(x[0] == q for x in per[p])}
              for p in order[:10]}
    # behaviour (main maps only, like the rest of the table)
    for p in range(n):
        g = [(r, r["i"] == p) for r in rows if p in (r["i"], r["j"])]
        if not g:
            continue
        mine = lambda r, first, k: r[k] if first else r[k + "_en"]   # noqa: E731
        won = [r for r, f in g if r["result"] == ("win" if f else "lose")]
        shots = sum(mine(r, f, "shots") for r, f in g)
        stats[p]["beh"] = dict(
            shots=shots / len(g),
            accuracy=sum(mine(r, f, "dmg") for r, f in g) / shots if shots else float("nan"),
            cancelled=sum(mine(r, f, "canc") for r, f in g) / len(g),
            win_destroyed=sum(r["end"] == "destroyed" for r in won) / len(won) if won else float("nan"),
            win_timeout=sum(r["end"] == "timeout" for r in won) / len(won) if won else float("nan"),
            steps=float(np.mean([r["steps"] for r, _ in g])),
            draws=sum(r["result"] is None for r, _ in g) / len(g),
            hidden=sum(mine(r, f, "hidden") for r, f in g) / max(1, sum(r["steps"] for r, _ in g)))
    # series against outside rivals: every A model vs every participant that is not A, README rule
    final_maps = random_maps or official_maps
    family_a = [p for p in order if kinds[p] == "model" and names[p].startswith("A-")]
    outside = [q for q in order if not names[q].startswith("A-")]
    for p in family_a:
        out = {}
        for q in outside:
            p_off = series_win_prob(table, p, q, official_maps or final_maps)
            p_rnd = series_win_prob(table, p, q, final_maps)
            games = [x for x in per[p] if x[0] == q]
            out[names[q]] = dict(p_official=float(p_off), p_random=float(p_rnd), mean=float((p_off + p_rnd) / 2),
                                 losses=sum(x[1] == "lose" for x in games), games=len(games))
        stats[p]["vs_outside"] = out
    return [stats[p] for p in order], suspicious, matrix


# ------------------------------------------------------------------------ report
def pct(x):
    return "-" if x is None or x != x else f"{100 * x:.0f}%"


def fmt_gen(s):
    return "-" if s["elo_gen"] is None else f"{pct(s['wr_gen'])} ({s['elo_gen']:.0f})"


def write_report(path, stats, suspicious, matrix, names, meta):
    L = [f"# Liga {meta['date']}", "",
         f"{len(names)} participantes · {meta['n_games']} partidas · mapas: {meta['maps_summary']} · "
         f"{meta['minutes']:.1f} min · {meta['procs']} procesos", "",
         "| # | participante | tipo | Elo | W-D-L | win rate | vs top-3 | peor emparejamiento | aleatorios | "
         "oficiales | gen (Elo gen) | empates | daño | sospechosas | P(campeón) | series a moneda |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in enumerate(stats, 1):
        worst = f"{s['worst']} ({pct(s['worst_score'])})" if s["worst"] else "-"
        L.append(f"| {k} | {s['name']} | {s['kind']} | {s['elo']:.0f} | {s['win']}-{s['draw']}-{s['lose']} | "
                 f"{pct(s['win_rate'])} | {pct(s['wr_top3'])} | {worst} | {pct(s['wr_random'])} | "
                 f"{pct(s['wr_official'])} | {fmt_gen(s)} | {pct(s['draw'] / max(1, s['games']))} | {s['avg_damage']:.2f} | "
                 f"{s.get('suspicious', 0)} | {pct(s.get('p_champion'))} | {pct(s.get('coin_series'))} |")
    L += ["", "Puntuación = (W + 0.5·D) / partidas. Peor emparejamiento = rival con menor puntuación. "
          "P(campeón) = bracket de 8 al mejor de 3 (rondas en oficiales, final en aleatorios), "
          f"{meta['sims']} sorteos. Regla del README: los empates no cuentan (se juega la siguiente combinación "
          "de mapa y lado, hasta agotarlas: 2 × mapas), después decide el daño total de la serie y, si es igual, "
          "una moneda. Series a moneda = % de las series de ese modelo decididas a cara o cruz "
          f"(en todo el cuadro: {pct(stats[0].get('coin_series_all'))}). Los mapas gen (map_gen.py) no entran en el Elo, el W-D-L ni el bracket: "
          "solo en la columna gen (win rate y Elo propio, centrado en la media del principal).", "", "## Cara a cara (top-10, puntuación de la fila contra la columna)", ""]
    cols = list(matrix)
    L.append("| | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for r in cols:
        L.append(f"| {r} | " + " | ".join(pct(matrix[r].get(c)) if c != r else "·" for c in cols) + " |")
    L += ["", "## Comportamiento (mapas principales, sin gen)", "",
          "| participante | disparos/partida | puntería | balas anuladas/partida | victorias por destrucción | "
          "victorias por timeout | duración media | empates | pasos escondido |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        b = s.get("beh")
        if b:
            L.append(f"| {s['name']} | {b['shots']:.1f} | {pct(b['accuracy'])} | {b['cancelled']:.2f} | "
                     f"{pct(b['win_destroyed'])} | {pct(b['win_timeout'])} | {b['steps']:.0f} | "
                     f"{pct(b['draws'])} | {pct(b['hidden'])} |")
    L += ["", "Puntería = impactos / disparos. Victorias por destrucción / timeout: sobre las victorias. "
          "Escondido = en un arbusto e invisible para el rival.", ""]
    fam = [s for s in stats if s.get("vs_outside")]
    if fam:
        L += ["", "## Series contra rivales de fuera (familia A)", "",
              "P(ganar la serie al mejor de 3 con la regla del cuadro) contra cada rival que no es A: en oficiales "
              "(rondas) y en aleatorios (final), y la media de las dos. Derrotas = partidas perdidas contra ese rival "
              f"en los mapas principales. {OUTSIDE_SERIES} series simuladas por casilla.", "",
              "| modelo A | media | peor rival (media) | rivales con media < 50 % | derrotas |", "|---|---|---|---|---|"]
        for s in fam:
            v = s["vs_outside"]
            worst = min(v, key=lambda q: v[q]["mean"])
            L.append(f"| {s['name']} | {pct(np.mean([x['mean'] for x in v.values()]))} | {worst} "
                     f"({pct(v[worst]['mean'])}) | {sum(x['mean'] < 0.5 for x in v.values())} | "
                     f"{sum(x['losses'] for x in v.values())} |")
        L += ["", "| modelo A | rival | oficiales | aleatorios | media | derrotas / partidas |",
              "|---|---|---|---|---|---|"]
        for s in fam:
            for q, x in s["vs_outside"].items():
                L.append(f"| {s['name']} | {q} | {pct(x['p_official'])} | {pct(x['p_random'])} | "
                         f"{pct(x['mean'])} | {x['losses']} / {x['games']} |")
    L += ["", f"## Partidas sospechosas (timeout con 0 daño): {len(suspicious)}", ""]
    for r in suspicious[:30]:
        L.append(f"- {names[r['i']]} (lado {r['side']}) vs {names[r['j']]} · {r['map']} · seed {r['seed']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# -------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], help="paths or globs of .zip models")
    ap.add_argument("--baselines", nargs="*", default=[], help="levels: easy medium hard")
    ap.add_argument("--bots", nargs="*", default=[], help="bot specs: basic, my:rusher, ...")
    ap.add_argument("--official", nargs="*", default=None, help="official maps (default: all)")
    ap.add_argument("--no-aug", action="store_true", help="skip the official_aug variants")
    ap.add_argument("--random", type=int, default=8, help="number of fixed-seed random maps")
    ap.add_argument("--gen-maps", type=int, default=0, help="extra maps from map_gen.py (own column)")
    ap.add_argument("--gen-seed", type=int, default=20261107, help="seed of the --gen-maps maps")
    ap.add_argument("--bot-seeds", type=int, default=3, help="games per (map, side) when a bot plays")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--sims", type=int, default=20000, help="bracket simulations")
    ap.add_argument("--save-all-replays", action="store_true",
                    help="write every game to <output>_replays.jsonl.gz (for winners_dataset.py)")
    ap.add_argument("--replays-dir", default=None, help="save replays of the losses of the top-3 models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    _worker_init()   # replays are played in this process
    t0 = time.time()

    specs, names, kinds = [], [], []
    paths = sorted({p.replace(os.sep, "/") for g in args.models for p in (glob.glob(g) or [g])})
    for p in paths:
        specs.append(f"model:{p}"), names.append(os.path.splitext(os.path.basename(p))[0]), kinds.append("model")
    for b in args.baselines:
        specs.append(f"baseline:{b}"), names.append(f"baseline:{b}"), kinds.append("baseline")
    for b in args.bots:
        specs.append(b), names.append(b), kinds.append("bot")
    if len(specs) < 2:
        sys.exit("Need at least 2 participants")
    assert len(set(names)) == len(names), f"duplicate names: {names}"

    official = args.official if args.official is not None else list_maps()
    official_keys = [f"official:{m}" for m in official]
    aug_keys = [] if args.no_aug else [f"aug:{os.path.relpath(p, ROOT)}" for p in _official_aug()
                                       if not p.endswith("__orig.txt")
                                       and os.path.basename(p).split("__")[0] in official]
    random_keys = [f"random:{RANDOM_SEED0 + k}" for k in range(args.random)]
    gen_keys = [f"gen:{args.gen_seed}:{k}" for k in range(args.gen_maps)]
    map_keys = official_keys + aug_keys + random_keys + gen_keys

    tasks = []
    for i, j in itertools.combinations(range(len(specs)), 2):
        seeds = range(1) if kinds[i] != "bot" and kinds[j] != "bot" else range(args.bot_seeds)
        for key in map_keys:
            for side in (0, 1):
                for s in seeds:
                    tasks.append((i, j, key, side, s, specs[i], specs[j]))
    print(f"[league] {len(specs)} participants, {len(map_keys)} maps "
          f"({len(official_keys)} official + {len(aug_keys)} aug + {len(random_keys)} random + {len(gen_keys)} gen), "
          f"{len(tasks)} games, {args.procs} procs", flush=True)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"league_{date}{'_' + args.tag if args.tag else ''}")
    replays = gzip.open(base + "_replays.jsonl.gz", "wt", encoding="utf-8") if args.save_all_replays else None

    rows = []
    with Pool(args.procs, initializer=_worker_init, initargs=(args.save_all_replays,)) as pool:
        for k, r in enumerate(pool.imap_unordered(play, tasks, chunksize=4), 1):
            if replays:
                rep = r.pop("replay")
                rep["league"] = dict(r, p0=names[r["i"] if r["side"] == 0 else r["j"]],
                                     p1=names[r["j"] if r["side"] == 0 else r["i"]])
                replays.write(json.dumps(rep) + "\n")
            rows.append(r)
            if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[league] {k}/{len(tasks)} games · {el / 60:.1f} min · "
                      f"ETA {el / k * (len(tasks) - k) / 60:.1f} min", flush=True)

    if replays:
        replays.close()
        print(f"[league] all replays in {base}_replays.jsonl.gz")
    stats, suspicious, matrix = analyse(names, kinds, rows, official_keys, random_keys, args.sims)
    meta = dict(date=date, n_games=len(rows), procs=args.procs, sims=args.sims, specs=specs,
                minutes=(time.time() - t0) / 60, maps=map_keys,
                maps_summary=f"{len(official_keys)} oficiales + {len(aug_keys)} aug + {len(random_keys)} aleatorios + {len(gen_keys)} gen",
                args=vars(args))
    write_report(base + ".md", stats, suspicious, matrix, names, meta)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, names=names, ranking=stats, matrix=matrix, games=rows), f)
    print(f"[league] wrote {base}.md / .json")
    for k, s in enumerate(stats[:10], 1):
        print(f"  {k:2d}. {s['name']:28s} Elo {s['elo']:6.0f}  wr {pct(s['win_rate']):>4s}  "
              f"top3 {pct(s['wr_top3']):>4s}  rnd {pct(s['wr_random']):>4s}  gen {fmt_gen(s):>10s}  champ {pct(s.get('p_champion'))}")

    if args.replays_dir:
        idx = {nm: p for p, nm in enumerate(names)}
        top = [idx[s["name"]] for s in stats if s["kind"] == "model"][:3]
        for r in rows:
            for p, lost in ((r["i"], r["result"] == "lose"), (r["j"], r["result"] == "win")):
                if p in top and lost:
                    res = play((r["i"], r["j"], r["map"], r["side"], r["seed"], specs[r["i"]], specs[r["j"]]),
                               full=True)
                    fn = f"{names[p]}__vs__{names[r['j'] if p == r['i'] else r['i']]}__" \
                         f"{r['map'].replace(':', '_').replace('/', '_')}__s{r['side']}_{r['seed']}.json"
                    save_replay(res, os.path.join(args.replays_dir, fn.replace(':', '_')))
        print(f"[league] replays of top-3 losses in {args.replays_dir}")


if __name__ == "__main__":
    main()
