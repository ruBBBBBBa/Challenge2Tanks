"""
Watch the league snapshots of a training run and evaluate each new one with eval_vs.py (PC2).

    python student/watch_snapshots.py --name D2-g1 --rivals baseline:hard ... models/reserved/R1.zip \
        --pid <training pid> --procs 3

Every --every seconds: new models/league/<name>_<k>k.zip (size unchanged for --settle s) -> eval_vs.py
(CPU, --procs workers) -> results/eval_<name>/<snapshot>.md/.json and one row in results/eval_<name>.md.
Stops when --pid is gone and every snapshot has been evaluated. R1 may be a rival: it only plays here.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def alive(pid):
    if not pid:
        return True
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def short(spec):
    s = spec.split(":", 1)[1] if spec.startswith("model:") else spec
    return os.path.basename(s).replace(".zip", "")


def row(snap, res):
    cells, tot = [], {"o": [0, 0, 0], "a": [0, 0, 0]}
    for rival, r in res.items():
        o, a = (list(r[k].values())[:3] for k in sorted(r, key=lambda k: not k.startswith("oficiales")))
        for t, v in (("o", o), ("a", a)):
            tot[t] = [x + y for x, y in zip(tot[t], v)]
        cells.append(f"{o[0]}-{o[1]}-{o[2]} · {a[0]}-{a[1]}-{a[2]}")
    fmt = lambda v: f"{v[0]}-{v[1]}-{v[2]}"  # noqa: E731
    return f"| {snap} | " + " | ".join(cells) + f" | {fmt(tot['o'])} | {fmt(tot['a'])} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rivals", nargs="+", required=True)
    ap.add_argument("--pid", type=int, default=0, help="training process; stop when it is gone")
    ap.add_argument("--pid-file", default=None, help="read the training pid from this file on every poll")
    ap.add_argument("--grace", type=int, default=900, help="keep waiting this long after the pid is gone")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--extra", nargs="*", default=[], help="models to evaluate first (e.g. the init)")
    ap.add_argument("--outbox", action="store_true", help="copy each evaluated snapshot to models/outbox/")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    outdir, summary = f"results/eval_{args.name}", f"results/eval_{args.name}.md"
    os.makedirs(outdir, exist_ok=True)
    done = {os.path.basename(p)[:-5] for p in glob.glob(f"{outdir}/*.json")}
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(f"# Evaluación de los snapshots de {args.name}\n\n"
                    "Determinista, con `student/eval_vs.py`: 4 mapas oficiales y 8 aleatorios de la liga, los dos lados. "
                    "Cada celda es «oficiales · aleatorios» en W-D-L. R1 solo como rival de evaluación.\n\n"
                    "| snapshot | " + " | ".join(short(r) for r in args.rivals) + " | total oficiales | total aleatorios |\n"
                    + "|---" * (len(args.rivals) + 3) + "|\n")
    seen, dead_since = {}, None
    while True:
        snaps = sorted(glob.glob(f"models/league/{args.name}_*k.zip"),
                       key=lambda p: int(re.search(r"_(\d+)k\.zip$", p).group(1)))
        todo = [p for p in args.extra if os.path.basename(p)[:-4] not in done] + \
               [p for p in snaps if os.path.basename(p)[:-4] not in done]
        if args.pid_file:                                   # re-read: watchdog.py may relaunch with a new pid
            try:
                args.pid = int(open(args.pid_file).read().strip())
            except (OSError, ValueError):
                pass
        running = alive(args.pid)
        if running:
            dead_since = None
        else:
            dead_since = dead_since or time.time()
            print(f"[watch] {time.strftime('%H:%M')} training pid {args.pid} is not running", flush=True)
            running = time.time() - dead_since < args.grace  # give watchdog.py time to relaunch it
        for p in todo:
            name = os.path.basename(p)[:-4]
            size = os.path.getsize(p)
            if p not in args.extra and (seen.get(p) != size or time.time() - os.path.getmtime(p) < args.settle):
                seen[p] = size                              # still being written
                continue
            print(f"[watch] {time.strftime('%H:%M')} evaluating {p}", flush=True)
            cmd = [sys.executable, "-W", "ignore", "student/eval_vs.py", "--models", p, "--rivals", *args.rivals,
                   "--procs", str(args.procs), "--out", f"{outdir}/{name}"]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"[watch] FAILED {p}:\n{r.stderr[-2000:]}", flush=True)
                continue
            with open(f"{outdir}/{name}.json", encoding="utf-8") as f:
                res = json.load(f)["results"]
            res = next(iter(res.values()))
            with open(summary, "a", encoding="utf-8") as f:
                f.write(row(name, res) + "\n")
            done.add(name)
            print(f"[watch] {row(name, res)}", flush=True)
            if args.outbox:                                 # hand it to sync_models.py --publish
                os.makedirs("models/outbox", exist_ok=True)
                shutil.copy2(p, "models/outbox/")
                print(f"[watch] copied {p} -> models/outbox/", flush=True)
        if not running and not [p for p in snaps if os.path.basename(p)[:-4] not in done]:
            print("[watch] training finished and every snapshot evaluated", flush=True)
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
