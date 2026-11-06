"""
Single training CLI (see CLAUDE.md section 6).

    python student/train_agent.py --name A-g1 --init baseline:hard --reward generalista --timesteps 5000000
    python student/train_agent.py --name A-g2 --init models/A-g1.zip --league-glob "models/league/*.zip" --pfsp variance
    python student/train_agent.py --name D-g1 --algo recurrent --init models/D-distill.zip --lr 3e-5

Saves models/<name>.zip + models/<name>.json, snapshots in models/league/<name>_<k>k.zip,
checkpoints in models/checkpoints_<name>/, TensorBoard in logs/<name>_*, and appends to BITACORA.md.
"""
import argparse
import datetime
import glob
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from student.envs import StudentTanksEnv  # noqa: E402
from student.pfsp import PFSPCallback, mix_weights  # noqa: E402
from tanks import WinRateCurriculum, evaluate_agent  # noqa: E402
from tanks.agents.base import BASELINES_DIR  # noqa: E402
from tanks.rewards import default_reward  # noqa: E402

try:
    from student.my_rewards import REWARDS
except ImportError:          # my_rewards.py (PC3) without REWARDS yet
    REWARDS = {}
REWARDS = {"default": default_reward, **REWARDS}

GEN1_MIX = {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10,
            "basic": 0.10, "aggressive": 0.10, "defensive": 0.10,
            "my:bushcamper": 0.075, "my:rusher": 0.075}

# hard.zip hyperparameters, used when training from scratch (--init none)
HARD_KWARGS = dict(n_epochs=10, gamma=0.995, gae_lambda=0.95, batch_size=2048)


class BlendCallback(BaseCallback):
    """reward_fn.blend goes 0 -> 1 linearly over `steps` timesteps."""

    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def _on_rollout_start(self):
        blend = min(1.0, self.num_timesteps / self.steps) if self.steps > 0 else 1.0
        self.training_env.env_method("set_reward_param", "blend", blend)
        self.logger.record("reward/blend", blend)

    def _on_step(self):
        return True


class SnapshotCallback(BaseCallback):
    def __init__(self, name, every):
        super().__init__()
        self.name, self.every, self.next = name, every, every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            path = os.path.join(ROOT, "models", "league", f"{self.name}_{self.next // 1000}k.zip")
            self.model.save(path)
            print(f"[snapshot] {os.path.relpath(path, ROOT)}")
            self.next += self.every
        return True


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--algo", choices=["ppo", "recurrent"], default="ppo")
    ap.add_argument("--init", default="baseline:hard", help="path .zip, baseline:<level> or none")
    ap.add_argument("--reward", default="default", choices=sorted(REWARDS))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--opponents", type=json.loads, default=None, help="JSON {spec: weight}")
    g.add_argument("--stages", type=json.loads, default=None, help="JSON list of curriculum stages")
    ap.add_argument("--league-glob", default=None, help='e.g. "models/league/*.zip"')
    ap.add_argument("--league-share", type=float, default=0.5)
    ap.add_argument("--pfsp", choices=["off", "variance", "hard"], default="variance")
    ap.add_argument("--maps", choices=["official", "official_aug", "random", "mixed", "gen", "mixed2"], default="mixed")
    ap.add_argument("--scenario-prob", type=float, default=0.0, help="probability of a prepared start")
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--clip-range", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blend-steps", type=int, default=300_000)
    ap.add_argument("--snapshot-every", type=int, default=250_000)
    ap.add_argument("--eval-games", type=int, default=20)
    return ap.parse_args(argv)


def init_path(init):
    if init.startswith("baseline:"):
        return os.path.join(BASELINES_DIR, init.split(":", 1)[1] + ".zip")
    return init


def build_model(args, env):
    if args.algo == "recurrent":
        from sb3_contrib import RecurrentPPO as Algo
        policy = "MlpLstmPolicy"
    else:
        Algo, policy = PPO, "MlpPolicy"
    n_steps = max(64, 8192 // args.n_envs)
    common = dict(n_steps=n_steps, ent_coef=args.ent_coef, seed=args.seed,
                  tensorboard_log=os.path.join(ROOT, "logs"))
    if args.init == "none":
        return Algo(policy, env, learning_rate=args.lr, clip_range=args.clip_range, device="cpu", verbose=1,
                    policy_kwargs=dict(net_arch=[256, 256]), **HARD_KWARGS, **common)
    model = Algo.load(init_path(args.init), env=env, device="cpu",
                      custom_objects={"learning_rate": args.lr, "clip_range": args.clip_range}, **common)
    model.verbose = 1
    return model


def fmt_eval(res):
    return "\n".join(f"| {o} | {r['win']} | {r['draw']} | {r['lose']} | {100 * r['win_rate']:.0f}% | "
                     f"{r['avg_damage']:.2f} |" for o, r in res.items())


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    os.makedirs("models/league", exist_ok=True)
    started = datetime.datetime.now()

    reward_fn = REWARDS[args.reward]
    reward_fn = reward_fn() if isinstance(reward_fn, type) else reward_fn
    has_blend = hasattr(reward_fn, "blend")

    fixed = args.opponents or GEN1_MIX
    league = sorted(glob.glob(args.league_glob)) if args.league_glob and not args.stages else []
    league = [f"model:{p.replace(os.sep, '/')}" for p in league]
    opponent = args.stages[0] if args.stages else mix_weights(fixed, league, args.league_share)

    env = make_vec_env(StudentTanksEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(opponent=opponent, maps=args.maps, reward_fn=reward_fn,
                                       scenario_prob=args.scenario_prob),
                       monitor_kwargs=dict(info_keywords=("is_success",)))
    model = build_model(args, env)
    lr_eff, clip_eff = model.lr_schedule(1.0), model.clip_range(1.0)
    assert abs(lr_eff - args.lr) < 1e-12 and abs(clip_eff - args.clip_range) < 1e-12, (lr_eff, clip_eff)
    print(f"[train] {args.name}: init={args.init} lr={lr_eff} clip={clip_eff} ent={model.ent_coef} "
          f"n_steps={model.n_steps}x{args.n_envs} batch={model.batch_size} reward={args.reward} "
          f"league={len(league)}")

    callbacks = [SnapshotCallback(args.name, args.snapshot_every),
                 CheckpointCallback(save_freq=max(500_000 // args.n_envs, 1),
                                    save_path=f"models/checkpoints_{args.name}/", name_prefix=args.name)]
    if has_blend:
        callbacks.append(BlendCallback(args.blend_steps))
    if args.stages:
        callbacks.append(WinRateCurriculum(args.stages))
    else:
        callbacks.append(PFSPCallback(fixed, league, args.league_share, args.pfsp))

    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.name)
    model.save(f"models/{args.name}")
    env.close()

    meta = dict(vars(args), steps=int(model.num_timesteps), started=started.isoformat(timespec="seconds"),
                finished=datetime.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
                pc=os.environ.get("PC", "PC1"), league=league, lr_effective=lr_eff)
    with open(f"models/{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved models/{args.name}.zip\n")

    res = evaluate_agent(f"models/{args.name}.zip", games=args.eval_games)
    with open("BITACORA.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {meta['finished'][:16].replace('T', ' ')} · {meta['pc']} · {args.name}\n\n"
                f"- Comando: `python student/train_agent.py {' '.join(sys.argv[1:] if argv is None else argv)}`\n"
                f"- Pasos: {meta['steps']:,} · inicio {meta['started'][:16]} · lr efectivo {lr_eff}\n"
                f"- Evaluación determinista ({args.eval_games} partidas/rival):\n\n"
                f"| rival | win | draw | lose | win rate | daño |\n|---|---|---|---|---|---|\n{fmt_eval(res)}\n\n"
                f"- Notas: \n")


if __name__ == "__main__":
    main()
