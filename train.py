"""
First training run: PPO against the "basic" heuristic bot.

    python train.py                       # 300k steps vs basic -> models/tanks_ppo.zip
    python train.py --timesteps 1000000 --n-envs 12 --name ppo_v1

Then:
    python tools/evaluate.py --agent models/tanks_ppo.zip
    python tools/play_match.py --p0 models/tanks_ppo.zip --p1 basic
    tensorboard --logdir logs

This script works out of the box. It uses the reward in student/my_rewards.py, so if you
change your reward, this script picks it up. When you want curriculum, opponent mixtures,
self-play or other algorithms, move on to student/train_advanced.py.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from student.my_rewards import my_reward
from tanks import TanksEnv, evaluate_agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8, help="parallel environments (one process each)")
    ap.add_argument("--opponent", default="basic", help="any agent spec (basic, aggressive, baseline:easy, ...)")
    ap.add_argument("--name", default="tanks_ppo", help="saved as models/<name>.zip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # paths relative to the challenge root

    env = make_vec_env(
        TanksEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(opponent=args.opponent, reward_fn=my_reward),
        monitor_kwargs=dict(info_keywords=("is_success",)),   # win rate in TensorBoard (rollout/success_rate)
    )
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", device="cpu", seed=args.seed)
    checkpoint = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                                    save_path=f"./models/checkpoints_{args.name}/", name_prefix=args.name)
    model.learn(total_timesteps=args.timesteps, callback=checkpoint, tb_log_name=args.name)
    model.save(f"./models/{args.name}")
    env.close()
    print(f"\nSaved models/{args.name}.zip\n")

    evaluate_agent(f"models/{args.name}.zip", games=10)


if __name__ == "__main__":   # required on Windows for SubprocVecEnv
    main()
