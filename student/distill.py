"""
Agent D: distill baseline:hard (MLP) into a RecurrentPPO MlpLstmPolicy.

    python student/distill.py gen   --steps 1500000 --n-envs 8          # hard plays, we record
    python student/distill.py train --epochs 30                         # KL (actor) + returns (critic)
    python tools/evaluate.py --agent models/D-distill.zip --games 40
    python student/train_agent.py --algo recurrent --init models/D-distill.zip ...   # then RL

gen: every episode hard plays deterministic or sampled (--p-det), against a mix of bots/hard/medium,
on "mixed" maps and both sides. Per step we store obs, hard's action probs, default reward and done.
train: truncated BPTT over segments of --seg-len steps, windows of --bptt steps, LSTM state carried
between windows (and reset at episode starts, as in inference).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from student.envs import StudentTanksEnv  # noqa: E402

HARD = "tanks/agents/baselines/hard.zip"
DATA = "models/checkpoints_D-distill/data.npz"
GAMMA = 0.995
GEN_OPPONENTS = {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.15, "baseline:medium": 0.15,
                 "basic": 0.15, "aggressive": 0.15, "defensive": 0.10, "my:bushcamper": 0.05}
# Hyperparameters saved in the zip for the RL phase (as hard, LR lower).
RL_KWARGS = dict(n_steps=1024, batch_size=2048, gamma=GAMMA, gae_lambda=0.95, n_epochs=10,
                 ent_coef=0.01, learning_rate=1e-4, clip_range=0.2)
POLICY_KWARGS = dict(lstm_hidden_size=256, n_lstm_layers=1, net_arch=dict(pi=[256], vf=[256]))


def make_env(seed, opponents=GEN_OPPONENTS, maps="mixed", scenario_prob=0.0):
    def _f():
        th.set_num_threads(1)
        env = StudentTanksEnv(opponent=opponents, maps=maps, scenario_prob=scenario_prob)
        env.reset(seed=seed)
        return env
    return _f


# ---------------------------------------------------------------------- gen
def gen(args):
    th.set_num_threads(2)
    teacher = PPO.load(args.teacher, device="cpu")  # any PPO MlpPolicy zip (hard by default)
    opponents = args.opponents or GEN_OPPONENTS
    # never a judge (R1, R2...) in the data
    banned = [x for x in [args.teacher, *opponents] if any(k in x for k in ("R1", "R2", "reserved", "eval_pc4"))]
    assert not banned, f"judges cannot be in the data: {banned}"
    print(f"privileged teacher: {args.privileged}")
    print(f"teacher {args.teacher} | maps {args.maps} | scenario_prob {args.scenario_prob} | opponents {opponents}")
    venv = SubprocVecEnv([make_env(args.seed + i, opponents, args.maps, args.scenario_prob)
                          for i in range(args.n_envs)])
    rng = np.random.default_rng(args.seed)
    E, L = args.n_envs, args.steps // args.n_envs
    obs_buf = np.zeros((L, E, 77), np.float32)
    prob_buf = np.zeros((L, E, 6), np.float32)
    rew_buf = np.zeros((L, E), np.float32)
    done_buf = np.zeros((L, E), bool)
    prob_off_buf = np.zeros((L, E, 6), np.float32) if args.privileged else None
    det = rng.random(E) < args.p_det
    obs = venv.reset()
    results, t0 = [], time.time()
    for t in range(L):
        with th.no_grad():
            # --privileged: the teacher sees the enemy even in a bush (StudentTanksEnv.privileged_obs); the
            # student keeps the official obs. probs_official = what the same teacher does without that info.
            tobs = np.array(venv.env_method("privileged_obs"), np.float32) if args.privileged else obs
            probs = teacher.policy.get_distribution(th.as_tensor(tobs)).distribution.probs.numpy()
            if args.privileged:
                prob_off_buf[t] = teacher.policy.get_distribution(th.as_tensor(obs)).distribution.probs.numpy()
        sampled = (probs.cumsum(1) > rng.random((E, 1))).argmax(1)
        act = np.where(det, probs.argmax(1), sampled)
        obs_buf[t], prob_buf[t] = obs, probs
        obs, rew, done, infos = venv.step(act)
        rew_buf[t], done_buf[t] = rew, done
        for e in np.flatnonzero(done):
            results.append((bool(det[e]), infos[e]["opponent"], infos[e]["result"]))
            det[e] = rng.random() < args.p_det
        if (t + 1) % 5000 == 0:
            n = (t + 1) * E
            print(f"{n} steps  {n / (time.time() - t0):.0f} steps/s  {len(results)} episodes", flush=True)
    venv.close()
    os.makedirs(os.path.dirname(args.data), exist_ok=True)
    extra = dict(probs_official=prob_off_buf, privileged=np.array(True)) if args.privileged else {}
    np.savez(args.data, obs=obs_buf, probs=prob_buf, rew=rew_buf, done=done_buf, teacher=np.array(args.teacher),
             **extra)
    print(f"saved {args.data}: {L * E} steps, {len(results)} episodes")
    for mode in (True, False):
        for opp in sorted({o for _, o, _ in results}):
            r = [res for d, o, res in results if d == mode and o == opp]
            if r:
                print(f"  {'det ' if mode else 'samp'} {opp:32s} n={len(r):4d} "
                      f"win={np.mean([x == 'win' for x in r]):.2f} draw={np.mean([x == 'draw' for x in r]):.2f}")


# -------------------------------------------------------------------- train
def returns_and_mask(rew, done):
    """Discounted returns (episode end = terminal: time is in the obs). Mask drops each env's unfinished tail."""
    ret = np.zeros_like(rew)
    mask = np.zeros_like(done)
    g = np.zeros(rew.shape[1], np.float32)
    seen = np.zeros(rew.shape[1], bool)
    for t in range(len(rew) - 1, -1, -1):
        g = rew[t] + GAMMA * g * ~done[t]
        seen |= done[t]
        ret[t], mask[t] = g, seen
    return ret, mask


def segments(x, seg_len):
    """(L, E, ...) -> (E * L // seg_len, seg_len, ...), each env stream cut in consecutive segments."""
    L = x.shape[0] // seg_len * seg_len
    x = np.swapaxes(x[:L], 0, 1)
    return x.reshape(-1, seg_len, *x.shape[2:])


def seq_forward(policy, obs, starts, states):
    """obs (S, T, 77), starts (S, T) -> logits (S*T, 6), values (S*T,), new states. Mirrors evaluate_actions."""
    S, T = starts.shape
    x = obs.reshape(S * T, -1)
    s = starts.reshape(S * T)
    lat_pi, st_pi = policy._process_sequence(x, states[0], s, policy.lstm_actor)
    lat_vf, st_vf = policy._process_sequence(x, states[1], s, policy.lstm_critic)
    logits = policy.action_net(policy.mlp_extractor.forward_actor(lat_pi))
    values = policy.value_net(policy.mlp_extractor.forward_critic(lat_vf)).squeeze(-1)
    return logits, values, (st_pi, st_vf)


def zero_states(policy, n):
    z = lambda: th.zeros(policy.lstm_actor.num_layers, n, policy.lstm_actor.hidden_size,  # noqa: E731
                         device=policy.device)
    return ((z(), z()), (z(), z()))


def run_epoch(policy, data, idx, args, opt=None):
    obs, probs, ret, mask, starts = data
    dev = policy.device
    tot = {"kl": 0.0, "acc": 0.0, "acc_conf": 0.0, "vloss": 0.0, "n": 0, "hid_hit": 0.0, "hid_n": 0.0}
    for b in range(0, len(idx), args.batch_seqs):
        ib = idx[b:b + args.batch_seqs]
        states = zero_states(policy, len(ib))
        for w in range(0, obs.shape[1], args.bptt):
            sl = slice(w, w + args.bptt)
            o, p, r, m, s = (th.as_tensor(a[ib, sl]).to(dev) for a in (obs, probs, ret, mask, starts))
            with th.set_grad_enabled(opt is not None):
                logits, v, states = seq_forward(policy, o, s.float(), states)
                p = p.reshape(-1, 6)
                logq = th.log_softmax(logits, -1)
                kl = (p * (th.log(p.clamp_min(1e-8)) - logq)).sum(-1).mean()
                m = m.reshape(-1).float()
                vloss = ((v - r.reshape(-1)) ** 2 * m).sum() / m.sum().clamp_min(1)
                loss = kl + args.vf_coef * vloss
                real = p.sum(-1) > 0.5                      # padding rows have p = 0
                if args.ce_coef:                            # cross-entropy with the teacher's favourite action
                    ce = -logq.gather(1, p.argmax(-1, keepdim=True)).squeeze(1)
                    loss = loss + args.ce_coef * (ce * real).sum() / real.sum().clamp_min(1)
                conf = real & (p.max(-1).values > 0.6)
                hit = (logits.argmax(-1) == p.argmax(-1)).float()
                hidden = real & (o.reshape(-1, o.shape[-1])[:, 8] == 0)   # enemy hidden in a bush
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
            states = tuple((h.detach(), c.detach()) for h, c in states)
            tot["kl"] += kl.item()
            tot["acc"] += hit.mean().item()
            tot["acc_conf"] += ((hit * conf).sum() / conf.sum().clamp_min(1)).item()
            tot["vloss"] += vloss.item()
            tot["hid_hit"] += (hit * hidden).sum().item()
            tot["hid_n"] += hidden.sum().item()
            tot["n"] += 1
    out = {k: tot[k] / tot["n"] for k in ("kl", "acc", "acc_conf", "vloss")}
    out["acc_hidden"] = tot["hid_hit"] / max(1.0, tot["hid_n"])
    out["n_hidden"] = int(tot["hid_n"])
    return out


def load_data(args):
    # Several data files: segments are cut per file (returns never flow from one file into the next) and
    # concatenated. Validation split per file: file k uses default_rng(seed + k), so file 0 keeps exactly
    # the split of a previous single-file run (no leakage when --init continues from that run).
    parts, val_idx, tr_idx, teachers, off = [], [], [], [], 0
    for k, path in enumerate(args.data):
        d = np.load(path)
        teachers.append(str(d["teacher"]) if "teacher" in d.files else HARD)   # older data files: hard
        ret, mask = returns_and_mask(d["rew"], d["done"])
        probs = d["probs"]
        mix = getattr(args, "priv_mix", 1.0)
        if mix < 1.0 and "probs_official" in d.files:   # --privileged data: blend with the teacher without vision
            probs = mix * probs + (1 - mix) * d["probs_official"]
            print(f"{path}: target = {mix} * privileged + {1 - mix:.2f} * official teacher probs")
        starts = np.ones_like(d["done"])
        starts[1:] = d["done"][:-1]
        seg = [segments(a, args.seg_len) for a in (d["obs"], probs, ret, mask, starts)]
        seg[4][:, 0] = True  # each segment starts from a zero LSTM state
        S = len(seg[0])
        perm = np.random.default_rng(args.seed + k).permutation(S)
        n_val = max(1, int(S * args.val_frac))
        val_idx.append(np.sort(perm[:n_val]) + off)
        tr_idx.append(perm[n_val:] + off)
        parts.append(seg)
        off += S
        print(f"{path}: teacher {teachers[-1]}, {S} segments x {args.seg_len} (val {n_val})")
        del d
    data = [np.concatenate([p[i] for p in parts]) for i in range(5)] if len(parts) > 1 else parts[0]
    del parts
    val_idx, tr_idx = np.concatenate(val_idx), np.concatenate(tr_idx)
    teacher = teachers[0] if len(set(teachers)) == 1 else ",".join(teachers)
    print(f"{len(data[0])} segments (train {len(tr_idx)}, val {len(val_idx)})")
    return data, tr_idx, val_idx, teacher


def train(args):
    from sb3_contrib import RecurrentPPO
    th.set_num_threads(args.threads)
    th.manual_seed(args.seed)
    data, tr_idx, val_idx, teacher = load_data(args)

    env = DummyVecEnv([lambda: StudentTanksEnv(opponent="basic", maps="official")])
    if args.init:
        model = RecurrentPPO.load(args.init, env=env, device=args.device)
        print(f"init from {args.init}")
    else:
        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=POLICY_KWARGS, seed=args.seed,
                             device=args.device, **RL_KWARGS)
    policy = model.policy
    print(f"device {policy.device}")
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    t_start, best, last = time.time(), None, 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        # cosine LR over --epochs, or over the time budget with --max-minutes
        frac = ep / args.epochs
        if args.max_minutes:
            frac = max(frac, (t0 - t_start) / (60 * args.max_minutes))
            if (t0 - t_start) + last > 60 * args.max_minutes:
                print(f"time budget reached ({args.max_minutes} min): stop before epoch {ep + 1}")
                break
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * min(frac, 1.0)))
        policy.train()
        tr = run_epoch(policy, data, np.random.permutation(tr_idx), args, opt)
        policy.eval()
        va = run_epoch(policy, data, val_idx, args)
        last = time.time() - t0
        print(f"epoch {ep + 1:3d}  lr {opt.param_groups[0]['lr']:.2e}  train kl {tr['kl']:.4f} acc {tr['acc']:.3f} "
              f"v {tr['vloss']:.3f} | val kl {va['kl']:.4f} acc {va['acc']:.3f} acc_conf {va['acc_conf']:.3f} "
              f"acc_hidden {va['acc_hidden']:.3f} v {va['vloss']:.3f}  ({last:.0f}s, {(time.time() - t_start) / 60:.0f} min)", flush=True)
        score = va["kl"] if args.select == "kl" else -va["acc"]
        if best is None or score < best:
            best = score
            model.save(args.out)
            print(f"  saved {args.out}")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({**vars(args), "best_val_kl" if args.select == "kl" else "best_val_acc": best if args.select == "kl" else -best, "policy_kwargs": POLICY_KWARGS, "rl_kwargs": RL_KWARGS,
                   "date": time.strftime("%Y-%m-%d %H:%M"), "computer": os.environ.get("TANKS_PC", "PC2"),
                   "init": "baseline:hard" if teacher == HARD else teacher, "teacher": teacher}, f, indent=2)


def agree(args):
    """Agreement with the teacher on the validation segments of --data (same split as train), overall and on the
    steps where the enemy is hidden in a bush. With --privileged data, also the memoryless teacher's own agreement
    (official obs vs privileged obs) on those steps: the bar a student without memory cannot beat."""
    from sb3_contrib import RecurrentPPO
    th.set_num_threads(args.threads)
    data, _, val_idx, teacher = load_data(args)
    for path in args.data:
        d = np.load(path)
        if "probs_official" in d.files:
            real = d["probs"].sum(-1) > 0.5
            hid = real & (d["obs"][..., 8] == 0)
            same = d["probs_official"].argmax(-1) == d["probs"].argmax(-1)
            print(f"{path}: teacher with the official obs vs privileged: all {same[real].mean():.3f} | "
                  f"hidden {same[hid].mean():.3f} ({hid.sum()} hidden steps of {real.sum()})")
    for m in args.models:
        model = RecurrentPPO.load(m, device="cpu")
        model.policy.eval()
        r = run_epoch(model.policy, data, val_idx, args)
        print(f"{m}: val acc {r['acc']:.3f} acc_conf {r['acc_conf']:.3f} acc_hidden {r['acc_hidden']:.3f} "
              f"({r['n_hidden']} hidden steps) kl {r['kl']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--steps", type=int, default=1_500_000)
    g.add_argument("--n-envs", type=int, default=8)
    g.add_argument("--p-det", type=float, default=0.5, help="share of episodes where the teacher plays deterministic")
    g.add_argument("--teacher", default=HARD, help="PPO MlpPolicy zip to imitate (default: hard)")
    g.add_argument("--opponents", type=json.loads, default=None, help="JSON {spec: weight} (default: GEN_OPPONENTS)")
    g.add_argument("--maps", default="mixed", help="map set for StudentTanksEnv (default: mixed)")
    g.add_argument("--scenario-prob", type=float, default=0.0, help="probability of a prepared start")
    g.add_argument("--privileged", action="store_true",
                   help="the teacher sees the enemy even when hidden in a bush; the student keeps the official obs")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--seg-len", type=int, default=2048)
    t.add_argument("--bptt", type=int, default=128)
    t.add_argument("--batch-seqs", type=int, default=128)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--out", default="models/D-distill.zip")
    t.add_argument("--init", default=None, help="RecurrentPPO zip to continue from (default: new network)")
    t.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    t.add_argument("--ce-coef", type=float, default=0.0,
                   help="weight of the cross-entropy with the teacher's favourite action (deterministic play)")
    t.add_argument("--select", default="kl", choices=["kl", "acc"], help="keep the epoch with best val kl or acc")
    t.add_argument("--max-minutes", type=float, default=0.0, help="time budget; the cosine LR follows it")
    t.add_argument("--data", nargs="+", default=[DATA], help="one or more gen files (concatenated)")
    t.add_argument("--priv-mix", type=float, default=1.0,
                   help="with --privileged data: target = mix * privileged + (1 - mix) * official teacher probs")
    g.add_argument("--data", default=DATA)
    ag = sub.add_parser("agree", help="agreement with the teacher on the val split, incl. hidden-enemy steps")
    ag.add_argument("--models", nargs="+", required=True)
    for k, v in dict(seg_len=2048, bptt=128, batch_seqs=128, threads=8).items():
        ag.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ag.add_argument("--val-frac", type=float, default=0.05)
    ag.add_argument("--data", nargs="+", default=[DATA])
    ag.add_argument("--vf-coef", type=float, default=0.5)
    ag.add_argument("--ce-coef", type=float, default=0.0)
    for p in (g, t, ag):
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"gen": gen, "train": train, "agree": agree}[a.cmd](a)
