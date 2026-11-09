"""
Report material in results/informe/ (PNG figures + informe.md with the tables). Trains nothing.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python student/report_figs.py [--procs 10] [--skip-games]

1. Elo and series against outside rivals per generation of A (from results/league_*_ligaK.json).
2. TensorBoard curves of A-g1..A-g19 one after another (results/tb/A-g*/): reward, episode length, win rate.
3. Exploiters B and C: their tables (results/exploiters_*.md) + what they found in A-g7_8000k
   (games replayed now with league.play: A-g7_8000k and A-g9 vs B/C g5-g7, deterministic).
4. D-g1b -> D2-distill -> D2b-distill on the common rivals (PC2's tables) + Elo of D2b in the last leagues.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_tb import ema, read_run  # noqa: E402
from league import (RANDOM_SEED0, ROOT, _official_aug, _worker_init, build_table, head_to_head,  # noqa: E402
                    list_maps, pct, play, series_win_prob)

OUT = os.path.join(ROOT, "results", "informe")
CANDIDATE = "A-g17_8000k"


def league_files():
    """[(K, path)] sorted by K, one per league number (liga1b replaces liga1: same models, fixed bracket)."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", "league_*_liga*.json")):
        m = re.search(r"_liga(\d+)(b?)\.json$", f)
        if m:
            k = int(m.group(1))
            if m.group(2) or k not in out:
                out[k] = f
    return sorted(out.items())


def gen_of(name):
    m = re.match(r"A-g(\d+)(?:_(\d+)k)?$", name)
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else None) if m else (None, None)


def outside_series(d, a, n=500):
    """Mean P(series) of A model `a` vs every non-A model/baseline of league d (bots left out), README rule,
    on official maps (rounds) and random maps (final), and its losses against them in the main maps."""
    names, specs, games = d["names"], d["meta"]["specs"], d["games"]
    off = [m for m in d["meta"]["maps"] if m.startswith("official:")]
    rnd = [m for m in d["meta"]["maps"] if m.startswith("random:")]
    main = set(m for m in d["meta"]["maps"] if not m.startswith("gen:"))
    table = build_table([g for g in games if g["map"] in main])
    i = names.index(a)
    rivals = [j for j, (nm, sp) in enumerate(zip(names, specs))
              if not nm.startswith("A-") and sp.startswith(("model:", "baseline:"))]
    p = [(series_win_prob(table, i, j, off, n) + series_win_prob(table, i, j, rnd, n)) / 2 for j in rivals]
    losses = sum(head_to_head(table, i, j, list(main))[2] for j in rivals)
    return float(np.mean(p)), losses, len(rivals)


# ------------------------------------------------------------------------- 1. Elo per generation
def elo_section():
    files = league_files()
    data = [(k, json.load(open(f))) for k, f in files]
    rows, per_league = [], []
    newest = {}
    for k, d in data:
        gens = [gen_of(s["name"])[0] for s in d["ranking"] if gen_of(s["name"])[0]]
        newest[k] = max(gens) if gens else None
        a = [s for s in d["ranking"] if gen_of(s["name"])[0]]
        hard = next((s["elo"] for s in d["ranking"] if s["name"] == "baseline:hard"), None)
        per_league.append((k, max(s["elo"] for s in a) if a else None, hard))
    for g in sorted({v for v in newest.values() if v}):
        k, d = [(k, d) for k, d in data if newest[k] == g][-1]          # last league where g is the newest A
        mine = [s for s in d["ranking"] if gen_of(s["name"])[0] == g]
        best = max(mine, key=lambda s: s["elo"])
        final = max(mine, key=lambda s: gen_of(s["name"])[1] or 10 ** 9)   # A-gN (no _k suffix) = final
        ps, losses, nr = outside_series(d, best["name"])
        rows.append(dict(g=g, liga=k, best=best["name"], elo_best=best["elo"], final=final["name"],
                         elo_final=final["elo"], series=ps, losses=losses, rivals=nr,
                         wr_random=best["wr_random"], draws=best["beh"]["draws"] if "beh" in best else None))
        print(f"[informe] A-g{g}: liga{k} {best['name']} Elo {best['elo']:.0f} series {ps:.0%} losses {losses}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    g = [r["g"] for r in rows]
    ax[0].plot(g, [r["elo_best"] for r in rows], "o-", label="mejor snapshot de la generación")
    ax[0].plot(g, [r["elo_final"] for r in rows], "s--", alpha=0.7, label="modelo final de la generación")
    ax[0].axhline(1500, color="grey", lw=1, ls=":", label="baseline:hard (1500)")
    ax[0].set(xlabel="generación de A", ylabel="Elo (Bradley–Terry, hard = 1500)",
              title="Elo de A por generación (en su liga)")
    ax[0].legend(fontsize=8)
    ax[1].plot(g, [100 * r["series"] for r in rows], "o-", color="tab:green", label="P(serie) media")
    ax[1].set(xlabel="generación de A", ylabel="P(ganar la serie) contra rivales de fuera (%)",
              title="Series contra rivales de fuera (no A, sin bots)")
    ax2 = ax[1].twinx()
    ax2.bar(g, [r["losses"] for r in rows], alpha=0.25, color="tab:red", label="derrotas")
    ax2.set_ylabel("derrotas contra ellos (partidas)")
    ax[1].legend(loc="lower right", fontsize=8), ax2.legend(loc="upper right", fontsize=8)
    for a in ax:
        a.set_xticks(g)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "elo_series_por_generacion.png"), dpi=130)
    plt.close(fig)

    L = ["## 1. Evolución de A por generación", "",
         "![Elo y series por generación](elo_series_por_generacion.png)", "",
         "Cada generación se mide en la última liga en la que es la generación de A más reciente. El Elo está "
         "anclado a `baseline:hard` = 1500, pero los rivales cambian de liga a liga: la comparación entre "
         "generaciones es aproximada. «Fuera» = participantes que no son A (B, C, D, E y los baselines; sin bots). "
         "P(serie) = media de oficiales (rondas) y aleatorios (final), regla del README, 500 series simuladas.", "",
         "| gen | liga | mejor snapshot | Elo | final | Elo final | P(serie) fuera | derrotas fuera | rivales | "
         "win rate aleatorios | empates |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['g']} | {r['liga']} | {r['best']} | {r['elo_best']:.0f} | {r['final']} | "
                 f"{r['elo_final']:.0f} | {pct(r['series'])} | {r['losses']} | {r['rivals']} | "
                 f"{pct(r['wr_random'])} | {pct(r['draws'])} |")
    return L


# ------------------------------------------------------------------------- 2. TensorBoard curves
TB_TAGS = [("rollout/ep_rew_mean", "recompensa media por episodio"),
           ("rollout/ep_len_mean", "longitud media del episodio (pasos)"),
           ("rollout/success_rate", "tasa de victorias en entrenamiento")]


def tb_section():
    runs = sorted((int(m.group(1)), p) for p in glob.glob(os.path.join(ROOT, "results", "tb", "A-g*"))
                  if (m := re.search(r"A-g(\d+)$", p)))
    fig, axes = plt.subplots(len(TB_TAGS), 1, figsize=(13, 9), sharex=True)
    offset, bounds, summary = 0, [], []
    for g, path in runs:
        run = read_run(path)
        last = max((s for t in run.values() for s, _ in t), default=0)
        for ax, (tag, _) in zip(axes, TB_TAGS):
            if run.get(tag):
                xs, ys = zip(*run[tag])
                ax.plot([(offset + x) / 1e6 for x in xs], ema(ys, 0.9), lw=1.2, color=f"C{g % 10}")
        summary.append((g, last, *[run[t][-1][1] if run.get(t) else None for t, _ in TB_TAGS]))
        bounds.append((offset / 1e6, g))
        offset += last
    for ax, (_, label) in zip(axes, TB_TAGS):
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        for x, g in bounds:
            ax.axvline(x, color="grey", lw=0.5, alpha=0.5)
    for x, g in bounds:
        axes[0].text(x, 1.02, f"g{g}", transform=axes[0].get_xaxis_transform(), fontsize=7)
    axes[-1].set_xlabel("pasos acumulados de la rama A (millones; cada franja es una generación)")
    fig.suptitle("Rama A · curvas de TensorBoard (EMA 0,9), generaciones una tras otra", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tb_rama_A.png"), dpi=130)
    plt.close(fig)
    L = ["", "## 2. Curvas de entrenamiento de A (TensorBoard)", "", "![Curvas de A](tb_rama_A.png)", "",
         "Generaciones una tras otra en pasos acumulados. Cada generación arranca del A elegido en la liga anterior "
         "(no siempre del final de la anterior), contra un pool distinto, así que los saltos en las fronteras son "
         "cambios de rivales, no olvido. `success_rate` lo registra `pfsp.py` (últimos 100 episodios).", "",
         "| gen | pasos | recompensa final | longitud final | victorias final |", "|---|---|---|---|---|"]
    for g, last, rew, ln, sr in summary:
        f = lambda v, s: "–" if v is None else s.format(v)  # noqa: E731
        L.append(f"| {g} | {last / 1e6:.1f}M | {f(rew, '{:.2f}')} | {f(ln, '{:.0f}')} | {f(sr, '{:.0%}')} |")
    return L


# ------------------------------------------------------------------------- 3. exploiters
EXPLOITERS = ["B-g5_3000k", "B-g6_4500k", "B-g7_6000k", "C-g5_3000k", "C-g6_3000k", "C-g7_4500k"]
TARGETS = {"A-g7_8000k": "models/candidatos/A-g7_8000k.zip", "A-g9": "models/candidatos/A-g9.zip"}


def exploiter_table(path):
    rows = [l for l in open(path, encoding="utf-8") if l.startswith("| ") and re.match(r"\| \d", l)]
    return [[c.strip() for c in l.strip().strip("|").split("|")] for l in rows]


def exploiters_section(procs, skip):
    L = ["", "## 3. Exploiters B y C (PC3)", "",
         "Entrenados contra el A líder (a partir de la generación 5, contra **A-g7_8000k**). Se publica el mejor "
         "contra A que gane ≥ 85 % a hard (tablas de `results/exploiters_B.md` y `_C.md`).", "",
         "| rama | gen | A rival | mejor snapshot | puntuación vs A | V-E-D vs A | vs hard | publicado |",
         "|---|---|---|---|---|---|---|---|"]
    for br in "BC":
        for c in exploiter_table(os.path.join(ROOT, "results", f"exploiters_{br}.md")):
            L.append(f"| {br} | {c[0]} | {c[2]} | {c[4]} | {c[5]} | {c[6]} | {c[8]} | {c[9]} |")
    if skip:
        return L
    maps = [f"official:{m}" for m in list_maps()] + \
           [f"aug:{os.path.relpath(p, ROOT)}" for p in _official_aug() if not p.endswith("__orig.txt")] + \
           [f"random:{RANDOM_SEED0 + k}" for k in range(8)]
    names = list(TARGETS) + EXPLOITERS
    specs = [f"model:{p}" for p in TARGETS.values()] + [f"model:models/league/{e}.zip" for e in EXPLOITERS]
    tasks = [(i, j, m, s, 0, specs[i], specs[j]) for i in range(len(TARGETS))
             for j in range(len(TARGETS), len(names)) for m in maps for s in (0, 1)]
    with Pool(procs, initializer=_worker_init, initargs=(False,)) as pool:
        rows = list(pool.imap_unordered(play, tasks, chunksize=2))
    by = defaultdict(list)
    for r in rows:
        by[(names[r["i"]], names[r["j"]])].append(r)
    L += ["", f"### Qué encontraron en A-g7_8000k ({len(rows)} partidas deterministas: 4 oficiales + 12 aug + "
          "8 aleatorios, los dos lados; A-g9 como contraste)", "",
          "| A | exploiter | W-D-L de A | derrotas de A por destrucción | daño de A / del exploiter | "
          "disparos A / exploiter | pasos medios en las derrotas | exploiter escondido |",
          "|---|---|---|---|---|---|---|---|"]
    lost_frac = {}
    for a in TARGETS:
        for e in EXPLOITERS:
            g = by[(a, e)]
            w = [sum(r["result"] == x for r in g) for x in ("win", None, "lose")]
            lost = [r for r in g if r["result"] == "lose"]
            lost_frac[(a, e)] = w[2] / len(g)
            lost_steps = f"{np.mean([r['steps'] for r in lost]):.0f}" if lost else "–"
            lost_ko = pct(sum(r["end"] == "destroyed" for r in lost) / len(lost)) if lost else "–"
            L.append(f"| {a} | {e} | {w[0]}-{w[1]}-{w[2]} | {lost_ko} | "
                     f"{np.mean([r['dmg'] for r in g]):.2f} / {np.mean([r['dmg_en'] for r in g]):.2f} | "
                     f"{np.mean([r['shots'] for r in g]):.1f} / {np.mean([r['shots_en'] for r in g]):.1f} | "
                     f"{lost_steps} | {pct(np.mean([r['hidden_en'] / max(1, r['steps']) for r in g]))} |")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(EXPLOITERS))
    for k, a in enumerate(TARGETS):
        ax.bar(x + (k - 0.5) * 0.38, [100 * lost_frac[(a, e)] for e in EXPLOITERS], 0.38, label=a)
    ax.set_xticks(x, EXPLOITERS, fontsize=8)
    ax.set(ylabel="% de partidas que pierde A", title="Exploiters contra A-g7_8000k (su objetivo) y A-g9")
    ax.legend(), ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exploiters_vs_A.png"), dpi=130)
    plt.close(fig)
    L += ["", "![Exploiters contra A](exploiters_vs_A.png)", "",
          "**Lectura.** Las derrotas de A-g7_8000k llegan tarde (de media, entre 290 y 500 pasos) y entre el 38 % y "
          "el 100 % son por timeout, no por destrucción. **B** (agresiva) dispara 2–3 veces más que A, mete un impacto y "
          "gana por HP al final. **C** (superviviente) apenas hace daño (0,2–0,8 por partida), pero con uno basta: "
          "C-g7 gana todas sus partidas a los 500 pasos por el desempate de HP/daño. El agujero de A-g7_8000k era "
          "conformarse con el empate y no responder a un primer impacto. Con B y C en el pool, A-g9 ya no pierde "
          "contra B y pierde 2 de 48 contra C-g6 y 2 de 48 contra C-g7."]
    return L


# ------------------------------------------------------------------------- 4. D lineage
# W-D-L copied from PC2's tables (BITACORA 2026-11-07 14:50; results/eval_2026-11-07_D_y_nuevos.md,
# results/eval_D2-distill.md, results/eval_D2b-distill.md). D-g1b: 40 games per rival (16 aug + 4 random x 2 sides);
# D2/D2b: 24 per rival (4 official + 8 random x 2 sides).
D_RIVALS = ["baseline:hard", "A-g6_4000k", "A-g5_24000k", "B-g4_3000k", "C-g4_6000k", "E-g1_2000k"]
D_DATA = {
    "D-g1b (hard destilado, checkpoint 5M de RL)": ["8-30-2", "2-0-38", "0-2-38", "4-12-24", "0-8-32", "0-0-40"],
    "D2-distill (de A-g6, sin RL)": ["10-10-4", "0-24-0", "0-22-2", "0-22-2", "2-22-0", "0-22-2"],
    "D2b-distill (+1,5M datos, CE)": ["22-2-0", "0-24-0", "18-6-0", "0-22-2", "2-22-0", "24-0-0"],
}


def d_section():
    parse = lambda s: tuple(map(int, s.split("-")))  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(D_RIVALS))
    L = ["", "## 4. Rama D (LSTM, PC2): D-g1b → D2-distill → D2b-distill", "",
         "Mismo agente RecurrentPPO; cambia el maestro de la destilación (hard → A-g6_4000k) y la cantidad de datos. "
         "W-D-L de las tablas de PC2 (D-g1b: 40 partidas por rival en aug + aleatorios; D2/D2b: 24 en oficiales + "
         "aleatorios). Puntuación = (W + 0,5·D) / partidas.", "",
         "| modelo | " + " | ".join(D_RIVALS) + " | puntuación | % derrotas |", "|---" * (len(D_RIVALS) + 3) + "|"]
    for k, (m, cells) in enumerate(D_DATA.items()):
        w = [parse(c) for c in cells]
        loss = [l / (a + b + l) for a, b, l in w]
        sc = [(a + 0.5 * b) / (a + b + l) for a, b, l in w]
        tot = np.sum(w, axis=0)
        L.append(f"| {m} | " + " | ".join(cells) + f" | {pct((tot[0] + 0.5 * tot[1]) / tot.sum())} | "
                 f"{pct(tot[2] / tot.sum())} |")
        ax[0].bar(x + (k - 1) * 0.27, [100 * v for v in loss], 0.27, label=m)
        ax[1].bar(x + (k - 1) * 0.27, [100 * v for v in sc], 0.27, label=m)
    for a, t in zip(ax, ["% de partidas perdidas", "puntuación (W + 0,5·D) %"]):
        a.set_xticks(x, D_RIVALS, fontsize=7, rotation=20)
        a.set_ylabel(t), a.grid(alpha=0.3, axis="y")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Rama D: del destilado de hard al destilado de A-g6")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "rama_D.png"), dpi=130)
    plt.close(fig)
    L += ["", "![Rama D](rama_D.png)", "", "Elo en las últimas ligas (hard = 1500):", "",
          "| liga | " + " | ".join(["D2b-distill", "mejor D2-g1", "mejor A"]) + " |", "|---|---|---|---|"]
    for k, f in league_files()[-4:]:
        d = json.load(open(f))
        e = {s["name"]: s["elo"] for s in d["ranking"]}
        d2 = max(((v, n) for n, v in e.items() if n.startswith("D2-g1")), default=None)
        a = max((v, n) for n, v in e.items() if n.startswith("A-"))
        L.append(f"| {k} | {e.get('D2b-distill', float('nan')):.0f} | "
                 f"{f'{d2[0]:.0f} ({d2[1]})' if d2 else '–'} | {a[0]:.0f} ({a[1]}) |")
    return L


# ------------------------------------------------------------------------- 5. D5 (privileged teacher)
def d5_section():
    agree_path = os.path.join(ROOT, "results", "D5_agree.txt")
    ev_path = os.path.join(ROOT, "results", "eval_D5.json")
    if not (os.path.exists(agree_path) and os.path.exists(ev_path)):
        return []
    txt = open(agree_path, encoding="utf-8").read()
    ceiling = float(re.search(r"hidden ([0-9.]+) \(", txt).group(1))
    acc = {m: (float(a), float(h)) for m, a, h in
           re.findall(r"models/(\S+)-distill\.zip: val acc ([0-9.]+) .*?acc_hidden ([0-9.]+)", txt)}
    ev = json.load(open(ev_path))
    names, bots = ev["names"], ev["bots"]
    bot_rows = ev["bot_games"]
    models = [n for n in names if n.endswith("distill") or n.startswith("A-")]
    loss = {(names[r["i"]], ev["bots"][r["j"] - len(names)]): [] for r in bot_rows}
    for r in bot_rows:
        loss[(names[r["i"]], bots[r["j"] - len(names)])].append(r["result"] == "lose")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["A-g17 sin ver\n(obs oficial)", "D2b-distill", "D5-distill"]
    vals = [ceiling, acc["D2b"][1], acc["D5"][1]]
    ax[0].bar(labels, [100 * v for v in vals], color=["grey", "tab:orange", "tab:green"])
    for k, v in enumerate(vals):
        ax[0].text(k, 100 * v + 1, f"{100 * v:.0f} %", ha="center")
    ax[0].set(ylabel="% acuerdo con A-g17 privilegiado", ylim=(0, 100),
              title="Pasos con el rival escondido en un arbusto")
    x = np.arange(len(bots))
    for k, m in enumerate(models):
        ax[1].bar(x + (k - 1) * 0.27, [100 * np.mean(loss[(m, b)]) for b in bots], 0.27, label=m)
    ax[1].set_xticks(x, bots)
    ax[1].set(ylabel="% de partidas perdidas", title="Contra bots que se esconden (44 mapas x 2 lados x 3 semillas)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D5_privilegiado.png"), dpi=130)
    plt.close(fig)
    L = ["", "## 5. D5: destilación con maestro privilegiado (PC1, solo CPU)", "",
         "El maestro A-g17_8000k ve al rival aunque esté en un arbusto (`distill.py gen --privileged`: `vector_obs` "
         "con el enemigo visible y su posición real). El alumno LSTM, iniciado desde D2b-distill, solo recibe la "
         "observación oficial. Son 1,5M pasos (rivales de D2b con `my:bushcamper` y `my:hitandhide` al 20 % cada uno; "
         "sin R1) y 60 min de KL + CE 0,5 en CPU (23 épocas). Evaluación en `results/eval_D5.md`.", "",
         "![D5](D5_privilegiado.png)", "",
         "| modelo | acuerdo total (val) | acuerdo con el rival escondido |", "|---|---|---|",
         f"| A-g17_8000k con la obs oficial (el propio maestro sin ver: referencia sin memoria) | – | {pct(ceiling)} |"]
    for m in ("D2b", "D5"):
        L.append(f"| {m}-distill | {pct(acc[m][0])} | {pct(acc[m][1])} |")
    L += ["", "| modelo | bot | % derrotas |", "|---|---|---|"]
    for m in models:
        for b in bots:
            L.append(f"| {m} | {b} | {pct(np.mean(loss[(m, b)]))} |")
    L += ["", "**Lectura.** Con el rival escondido, A-g17 hace otra cosa en el 75 % de los pasos si lo ve: la "
          "información oculta importa mucho. D5 coincide con el maestro privilegiado en el 62 % de esos pasos, frente "
          "al 25 % del propio maestro sin verlo y al 21 % de D2b. La LSTM sí saca algo de la memoria (dónde se escondió "
          "el rival, de dónde vienen las balas). Ojo: no se ha entrenado un MLP con los mismos datos, que sería el "
          "control sin memoria limpio. Parte de la mejora puede venir de imitar mejor al maestro, sin memoria. **En juego no compensa**: "
          "D5 pierde el 15–17 % contra los bots que se esconden (A-g17 el 0–1 % y D2b el 4–5 %), porque imita acciones "
          "que dependen de algo que no ve (el clásico hueco de imitación con maestro privilegiado). En la liga "
          "mejora a D2b (Elo 1780 frente a 1743, 10 derrotas frente a 60) sin llegar a A-g17 (1839, 0 derrotas)."]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--skip-games", action="store_true", help="do not replay the exploiter games")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    L = ["# Material para el informe · PC1", "",
         "Generado por `student/report_figs.py` (no entrena nada). Todas las ramas parten de `baseline:hard` "
         "(con permiso del profesor). Partidas deterministas, como en el torneo.", ""]
    L += elo_section() + tb_section() + exploiters_section(args.procs, args.skip_games) + d_section() + d5_section()
    with open(os.path.join(OUT, "informe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[informe] wrote {OUT}/informe.md")


if __name__ == "__main__":
    main()
