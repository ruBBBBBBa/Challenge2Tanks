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


def make_env(seed):
    def _f():
        th.set_num_threads(1)
        env = StudentTanksEnv(opponent=GEN_OPPONENTS, maps="mixed")
        env.reset(seed=seed)
        return env
    return _f


# ---------------------------------------------------------------------- gen
def gen(args):
    th.set_num_threads(2)
    teacher = PPO.load(HARD, device="cpu")
    venv = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    rng = np.random.default_rng(args.seed)
    E, L = args.n_envs, args.steps // args.n_envs
    obs_buf = np.zeros((L, E, 77), np.float32)
    prob_buf = np.zeros((L, E, 6), np.float32)
    rew_buf = np.zeros((L, E), np.float32)
    done_buf = np.zeros((L, E), bool)
    det = rng.random(E) < args.p_det
    obs = venv.reset()
    results, t0 = [], time.time()
    for t in range(L):
        with th.no_grad():
            probs = teacher.policy.get_distribution(th.as_tensor(obs)).distribution.probs.numpy()
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
    np.savez(args.data, obs=obs_buf, probs=prob_buf, rew=rew_buf, done=done_buf)
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
    z = lambda: th.zeros(policy.lstm_actor.num_layers, n, policy.lstm_actor.hidden_size)  # noqa: E731
    return ((z(), z()), (z(), z()))


def run_epoch(policy, data, idx, args, opt=None):
    obs, probs, ret, mask, starts = data
    tot = {"kl": 0.0, "acc": 0.0, "vloss": 0.0, "n": 0}
    for b in range(0, len(idx), args.batch_seqs):
        ib = idx[b:b + args.batch_seqs]
        states = zero_states(policy, len(ib))
        for w in range(0, obs.shape[1], args.bptt):
            sl = slice(w, w + args.bptt)
            o, p, r, m, s = (th.as_tensor(a[ib, sl]) for a in (obs, probs, ret, mask, starts))
            with th.set_grad_enabled(opt is not None):
                logits, v, states = seq_forward(policy, o, s.float(), states)
                p = p.reshape(-1, 6)
                logq = th.log_softmax(logits, -1)
                kl = (p * (th.log(p.clamp_min(1e-8)) - logq)).sum(-1).mean()
                m = m.reshape(-1).float()
                vloss = ((v - r.reshape(-1)) ** 2 * m).sum() / m.sum().clamp_min(1)
                loss = kl + args.vf_coef * vloss
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
            states = tuple((h.detach(), c.detach()) for h, c in states)
            tot["kl"] += kl.item()
            tot["acc"] += (logits.argmax(-1) == p.argmax(-1)).float().mean().item()
            tot["vloss"] += vloss.item()
            tot["n"] += 1
    return {k: tot[k] / tot["n"] for k in ("kl", "acc", "vloss")}


def train(args):
    from sb3_contrib import RecurrentPPO
    th.set_num_threads(args.threads)
    th.manual_seed(args.seed)
    d = np.load(args.data)
    ret, mask = returns_and_mask(d["rew"], d["done"])
    starts = np.ones_like(d["done"])
    starts[1:] = d["done"][:-1]
    data = [segments(a, args.seg_len) for a in (d["obs"], d["probs"], ret, mask, starts)]
    data[4][:, 0] = True  # each segment starts from a zero LSTM state
    S = len(data[0])
    perm = np.random.default_rng(args.seed).permutation(S)
    n_val = max(1, int(S * args.val_frac))
    val_idx, tr_idx = np.sort(perm[:n_val]), perm[n_val:]
    print(f"{S} segments x {args.seg_len} steps  (train {len(tr_idx)}, val {n_val})")

    env = DummyVecEnv([lambda: StudentTanksEnv(opponent="basic", maps="official")])
    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=POLICY_KWARGS, seed=args.seed,
                         device="cpu", **RL_KWARGS)
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = None
    for ep in range(args.epochs):
        t0 = time.time()
        policy.train()
        tr = run_epoch(policy, data, np.random.permutation(tr_idx), args, opt)
        policy.eval()
        va = run_epoch(policy, data, val_idx, args)
        sched.step()
        print(f"epoch {ep + 1:3d}  train kl {tr['kl']:.4f} acc {tr['acc']:.3f} v {tr['vloss']:.3f} | "
              f"val kl {va['kl']:.4f} acc {va['acc']:.3f} v {va['vloss']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if best is None or va["kl"] < best:
            best = va["kl"]
            model.save(args.out)
            print(f"  saved {args.out}")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({**vars(args), "best_val_kl": best, "policy_kwargs": POLICY_KWARGS, "rl_kwargs": RL_KWARGS,
                   "date": time.strftime("%Y-%m-%d %H:%M"), "computer": "PC2", "init": "baseline:hard"}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--steps", type=int, default=1_500_000)
    g.add_argument("--n-envs", type=int, default=8)
    g.add_argument("--p-det", type=float, default=0.5, help="share of episodes where hard plays deterministic")
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
    for p in (g, t):
        p.add_argument("--data", default=DATA)
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"gen": gen, "train": train}[a.cmd](a)
