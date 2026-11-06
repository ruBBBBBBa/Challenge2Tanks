"""
Generation chainer: wait for a training -> copy its curves -> league -> pick the best A -> next generation.

    nohup .venv/bin/python student/pipeline.py --wait-pid logs/A-g2.pid > logs/pipeline.out 2>&1 &
    python student/pipeline.py --wait-pid logs/A-g2.pid --dry-run       # only print what it would do
    touch STOP_PIPELINE                                                  # stop after the current step

Per generation (A-g2 -> liga2 -> A-g3 -> liga3 -> A-g4 with --generations 2):
 1. wait for the PID in --wait-pid (the running training);
 2. copy logs/<ID>_<n>/ (latest n) to results/tb/<ID>/;
 3. models/ligaK/: snapshots of <ID> every 4M steps + the last one, top-3 models of the previous league,
    every snapshot of other branches in models/league/ (B-*, C-*, D-*, X1-*). Never models/reserved/;
 4. league.py on models/ligaK/*.zip (baselines, the 6 bots of league 1, --gen-maps 8, --save-all-replays);
 5. pick the branch-A model with the best main Elo (tie: best P(champion));
 6. launch the next generation detached (nohup), unless another train_agent.py is running.
Log: logs/pipeline.log. Summary per league: results/pipeline_ligaK.md (+ an entry in BITACORA.md).
No git operations: the folder is shared, push by hand.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OTHER_BRANCHES = ("B-", "C-", "D-", "X1-")
BOTS = ["basic", "aggressive", "defensive", "my:rusher", "my:bushcamper", "my:hitandhide"]
OPPONENTS = {"baseline:hard": 0.20, "my:stoch:baseline:hard": 0.15, "baseline:medium": 0.15, "basic": 0.05,
             "aggressive": 0.05, "defensive": 0.05, "my:rusher": 0.05, "my:bushcamper": 0.15,
             "my:hitandhide": 0.15}
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DRY = False   # --dry-run: print only, write nothing (not even logs/pipeline.log)


class Stop(Exception):
    pass


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if DRY:
        return
    with open(os.path.join(ROOT, "logs", "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_stop(step):
    if os.path.exists(os.path.join(ROOT, "STOP_PIPELINE")):
        raise Stop(f"STOP_PIPELINE found after: {step}")


def alive(pid):
    try:   # a training we launched ourselves stays a zombie (kill(pid, 0) succeeds) until reaped
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_trainings():
    out = subprocess.run(["pgrep", "-af", "student/train_agent.py"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if "pgrep" not in ln and str(os.getpid()) != ln.split()[0]]


def next_id(run_id):
    m = re.fullmatch(r"(.*-g)(\d+)", run_id)
    if not m:
        raise ValueError(f"Cannot increment generation of {run_id!r} (expected <branch>-g<N>)")
    return f"{m.group(1)}{int(m.group(2)) + 1}"


def kilo(path):
    m = re.search(r"_(\d+)k\.zip$", path)
    return int(m.group(1)) if m else -1


# ------------------------------------------------------------------ steps
def wait_for(pid_file, poll, dry):
    if not os.path.exists(pid_file):
        log(f"WARNING: {pid_file} does not exist, assuming the training has finished")
        return
    pid = int(open(pid_file).read().split()[0])
    if dry:
        log(f"[dry] would wait for PID {pid} ({'running' if alive(pid) else 'already finished'})")
        return
    log(f"waiting for PID {pid} ({pid_file})")
    while alive(pid):
        check_stop(f"waiting for PID {pid}")
        time.sleep(poll)
    log(f"PID {pid} finished")


def copy_curves(run_id, dry):
    runs = sorted(glob.glob(os.path.join(ROOT, "logs", f"{run_id}_*")), key=lambda p: int(p.rsplit("_", 1)[1])
                  if p.rsplit("_", 1)[1].isdigit() else -1)
    runs = [p for p in runs if os.path.isdir(p) and p.rsplit("_", 1)[1].isdigit()]
    if not runs:
        log(f"WARNING: no logs/{run_id}_<n>/ to copy")
        return
    dst = os.path.join(ROOT, "results", "tb", run_id)
    log(f"{'[dry] would copy' if dry else 'copy'} {os.path.relpath(runs[-1], ROOT)} -> results/tb/{run_id}/")
    if not dry:
        shutil.copytree(runs[-1], dst, dirs_exist_ok=True)


def previous_league():
    files = [f for f in glob.glob(os.path.join(ROOT, "results", "league_*.json"))]
    return max(files, key=os.path.getmtime) if files else None


def participants(run_id, every):
    """[(source path, reason)] for models/ligaK/. Never anything under models/reserved/."""
    own = sorted(glob.glob(os.path.join(ROOT, "models", "league", f"{run_id}_*k.zip")), key=kilo)
    picked = [(p, f"{run_id} cada {every // 1_000_000}M") for p in own if kilo(p) * 1000 % every == 0]
    final = os.path.join(ROOT, "models", f"{run_id}.zip")
    if os.path.exists(final) and os.path.exists(final[:-4] + ".json"):   # final == last snapshot: keep one
        steps = json.load(open(final[:-4] + ".json", encoding="utf-8")).get("steps", -1)
        picked = [(p, w) for p, w in picked if kilo(p) * 1000 != steps]
    last = final if os.path.exists(final) else (own[-1] if own else None)
    if last and last not in [p for p, _ in picked]:
        picked.append((last, f"{run_id} último"))

    prev = previous_league()
    if prev:
        d = json.load(open(prev, encoding="utf-8"))
        spec = dict(zip(d["names"], d["meta"]["specs"]))
        for s in [s for s in d["ranking"] if s["kind"] == "model"][:3]:
            picked.append((os.path.join(ROOT, spec[s["name"]][len("model:"):]),
                           f"top-3 de {os.path.basename(prev)} (Elo {s['elo']:.0f})"))
    for p in sorted(glob.glob(os.path.join(ROOT, "models", "league", "*.zip"))):
        if os.path.basename(p).startswith(OTHER_BRANCHES):
            picked.append((p, "otra rama"))

    out, names = [], set()
    for p, why in picked:
        name = os.path.basename(p)
        if "reserved" in os.path.relpath(p, ROOT).split(os.sep) or name in names:
            continue
        names.add(name)
        out.append((p, why))
    return out


def build_league_dir(k, parts, dry):
    dst = os.path.join(ROOT, "models", f"liga{k}")
    for p, why in parts:
        log(f"  {'[dry] ' if dry else ''}liga{k} <- {os.path.relpath(p, ROOT)} ({why})")
    if dry:
        return
    if os.path.exists(dst) and os.listdir(dst):
        raise RuntimeError(f"{os.path.relpath(dst, ROOT)} already exists and is not empty: refusing to overwrite")
    os.makedirs(dst, exist_ok=True)
    for p, _ in parts:
        shutil.copy2(p, dst)


def run_league(k, args):
    cmd = [PY, "student/league.py", "--models", f"models/liga{k}/*.zip", "--baselines", "hard", "medium",
           "--bots", *BOTS, "--bot-seeds", "1", "--gen-maps", "8", "--save-all-replays",
           "--procs", str(args.league_procs), "--tag", f"liga{k}", *shlex.split(args.league_extra)]
    if args.nice:
        cmd = ["nice", "-n", str(args.nice)] + cmd
    log(f"{'[dry] would run' if args.dry_run else 'run'}: {' '.join(shlex.quote(c) for c in cmd)}")
    if args.dry_run:
        return None, cmd
    with open(os.path.join(ROOT, "logs", f"liga{k}.out"), "w") as out:
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **THREADS}, stdout=out, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"league.py failed (exit {rc}), see logs/liga{k}.out")
    res = [f for f in glob.glob(os.path.join(ROOT, "results", f"league_*_liga{k}.json"))]
    return max(res, key=os.path.getmtime), cmd


def choose(result_file, branch="A-"):
    d = json.load(open(result_file, encoding="utf-8"))
    spec = dict(zip(d["names"], d["meta"]["specs"]))
    cands = [s for s in d["ranking"] if s["kind"] == "model" and s["name"].startswith(branch)]
    if not cands:
        raise RuntimeError(f"No branch-{branch} model in {result_file}")
    best = max(cands, key=lambda s: (round(s["elo"], 6), s.get("p_champion") or 0.0))
    tie = [s for s in cands if round(s["elo"], 6) == round(best["elo"], 6)]
    why = (f"mayor Elo principal de la rama A ({best['elo']:.0f})" +
           (f", desempate por P(campeón) {100 * (best.get('p_champion') or 0):.1f}%" if len(tie) > 1 else ""))
    return spec[best["name"]][len("model:"):], best, d, why


def train_cmd(new_id, init, k, pfsp, args):
    cmd = [PY, "student/train_agent.py", "--name", new_id, "--init", init,
           "--league-glob", f"models/liga{k}/*.zip", "--league-share", "0.5", "--pfsp", pfsp,
           "--reward", "default", "--lr", "1e-4", "--clip-range", "0.15", "--ent-coef", "0.02",
           "--maps", "mixed2", "--scenario-prob", "0.3", "--timesteps", str(args.timesteps),
           "--snapshot-every", "2000000", "--eval-games", "2", "--n-envs", "10",
           "--opponents", json.dumps(OPPONENTS, separators=(",", ":")), *shlex.split(args.train_extra)]
    return (["nice", "-n", str(args.nice)] if args.nice else []) + cmd


def launch(new_id, cmd, args):
    shown = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup " + " ".join(shlex.quote(c) for c in cmd) + \
            f" > logs/{new_id}.out 2>&1 &"
    busy = [] if args.ignore_running else running_trainings()
    if busy:
        log("NOT LAUNCHED: another train_agent.py is running:\n    " + "\n    ".join(busy))
        log(f"pending command: {shown}")
        return None, shown
    if args.dry_run:
        log(f"[dry] would launch: {shown}")
        return None, shown
    out = open(os.path.join(ROOT, "logs", f"{new_id}.out"), "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **THREADS}, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)   # survives the pipeline
    with open(os.path.join(ROOT, "logs", f"{new_id}.pid"), "w") as f:
        f.write(f"{proc.pid}\n")
    log(f"launched {new_id}: PID {proc.pid} -> logs/{new_id}.out, logs/{new_id}.pid")
    log(f"command: {shown}")
    return proc.pid, shown


def write_summary(k, parts, league_cmd, result_file, d, chosen, why, launched, shown, run_id, new_id):
    L = [f"# Pipeline · liga{k} · {now()}", "",
         f"Generación terminada: **{run_id}**. Siguiente: **{new_id}**.", "",
         "## Participantes (modelos)", ""]
    L += [f"- `{os.path.basename(p)}`: {why_}" for p, why_ in parts]
    L += ["- `baseline:hard`, `baseline:medium` y los bots: " + ", ".join(f"`{b}`" for b in BOTS), "",
          "## Liga", "", f"Resultados: `{os.path.relpath(result_file, ROOT)}` (y `.md`).", "",
          f"`{' '.join(shlex.quote(c) for c in league_cmd)}`", "",
          "| # | participante | Elo | win rate | vs top-3 | aleatorios | gen | P(campeón) |",
          "|---|---|---|---|---|---|---|---|"]
    pct = lambda x: "-" if x is None or x != x else f"{100 * x:.0f}%"   # noqa: E731
    for n, s in enumerate(d["ranking"][:12], 1):
        L.append(f"| {n} | {s['name']} | {s['elo']:.0f} | {pct(s['win_rate'])} | {pct(s['wr_top3'])} | "
                 f"{pct(s['wr_random'])} | {pct(s.get('wr_gen'))} | {pct(s.get('p_champion'))} |")
    L += ["", "## Elegido", "", f"`{chosen}`: {why}.", "", "## Lanzamiento", "",
          (f"Lanzado (PID {launched})." if launched else "**No lanzado** (ver `logs/pipeline.log`)."), "",
          f"```\n{shown}\n```", ""]
    path = os.path.join(ROOT, "results", f"pipeline_liga{k}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(ROOT, "BITACORA.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## {now()[:16]} · PC1 · pipeline: {run_id} → liga{k} → {new_id}\n\n"
                f"- Liga: `{os.path.relpath(result_file, ROOT)}` ({len(parts)} modelos + baselines + bots).\n"
                f"- Elegido: `{chosen}` ({why}).\n"
                f"- {'Lanzado ' + new_id + f' (PID {launched})' if launched else 'NO lanzado: había otro entrenamiento'}: "
                f"`{shown}`\n- Resumen: `results/pipeline_liga{k}.md`.\n")
    log(f"summary: results/pipeline_liga{k}.md")


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-pid", default="logs/A-g2.pid", help="pid file of the running training (logs/<ID>.pid)")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--pfsp-modes", default="variance,hard", help="one per chained generation (last one repeats)")
    ap.add_argument("--timesteps", type=int, default=30_000_000)
    ap.add_argument("--snapshot-step", type=int, default=4_000_000, help="league snapshots of the finished run")
    ap.add_argument("--league-procs", type=int, default=10)
    ap.add_argument("--poll", type=int, default=60, help="seconds between PID checks")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--nice", type=int, default=0, help="niceness of the league and the launched trainings")
    ap.add_argument("--league-extra", default="", help="extra league.py args (tests)")
    ap.add_argument("--train-extra", default="", help="extra train_agent.py args (tests)")
    ap.add_argument("--ignore-running", action="store_true", help="tests only: skip the running-training check")
    args = ap.parse_args(argv)
    global DRY
    DRY = args.dry_run
    os.chdir(ROOT)
    modes = args.pfsp_modes.split(",")
    pid_file = args.wait_pid
    run_id = os.path.splitext(os.path.basename(pid_file))[0]
    log(f"pipeline start{' (dry run)' if args.dry_run else ''}: after {run_id}, {args.generations} generations")
    try:
        for gen in range(args.generations):
            new_id = next_id(run_id)
            existing = [int(m.group(1)) for d in glob.glob("models/liga*")
                        if (m := re.fullmatch(r"models/liga(\d+)", d.replace(os.sep, "/")))]
            k = max(existing, default=0) + 1 + (gen if args.dry_run else 0)
            log(f"=== generation {gen + 1}/{args.generations}: {run_id} -> liga{k} -> {new_id}")
            wait_for(pid_file, args.poll, args.dry_run)
            check_stop("wait")
            copy_curves(run_id, args.dry_run)
            check_stop("copy curves")
            parts = participants(run_id, args.snapshot_step)
            build_league_dir(k, parts, args.dry_run)
            check_stop("league folder")
            result_file, league_cmd = run_league(k, args)
            check_stop("league")
            pfsp = modes[min(gen, len(modes) - 1)]
            if result_file is None:   # dry run
                chosen, why, d = f"models/liga{k}/<mejor A de liga{k}>", "(se decide con la liga)", None
            else:
                path, best, d, why = choose(result_file)
                chosen = os.path.relpath(path, ROOT).replace(os.sep, "/")
                log(f"chosen: {chosen}: {why}")
            launched, shown = launch(new_id, train_cmd(new_id, chosen, k, pfsp, args), args)
            if d is not None:
                write_summary(k, parts, league_cmd, result_file, d, chosen, why, launched, shown, run_id, new_id)
            if not launched and not args.dry_run:
                log("stopping: the next generation was not launched")
                return
            run_id, pid_file = new_id, f"logs/{new_id}.pid"
            check_stop("launch")
        log("pipeline finished")
    except Stop as e:
        log(str(e))


if __name__ == "__main__":
    main()
