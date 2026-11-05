"""
Advanced training template: algorithm, hyperparameters, curriculum, opponent mixtures, self-play.

    python student/train_advanced.py

Edit the CONFIG section. Typical workflow (manual self-play):
    1. NAME="v1", STAGES = curriculum against the bots            -> models/v1.zip
    2. NAME="v2", INIT_MODEL="models/v1.zip",
       STAGES = [{"model:models/v1.zip": 0.5, "basic": 0.25, "defensive": 0.25}]  -> models/v2.zip
    3. ... v3 against v2 + v1 + bots, and so on.

Always check the result with:  python tools/evaluate.py --agent models/<NAME>.zip
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from stable_baselines3 import A2C, DQN, PPO  # noqa: E402,F401
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from student.my_rewards import my_reward  # noqa: E402
from tanks import TanksEnv, WinRateCurriculum, evaluate_agent  # noqa: E402

# ============================================================================
# CONFIG  (TODO: everything here is yours to change)
# ============================================================================
NAME = "v1"                      # -> models/<NAME>.zip, TensorBoard run "<NAME>"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
SEED = 0

# Continue from an existing model (fine-tuning / next self-play generation), or None
INIT_MODEL = None                # e.g. "models/v1.zip"

# Curriculum: one stage per element. Each stage is an agent spec or a mixture {spec: weight}.
# It moves to the next stage when the win rate over the last WINDOW episodes >= THRESHOLD.
# Specs: "random", "basic", "aggressive", "defensive", "param:...", "baseline:easy|medium|hard",
#        "model:models/v1.zip", "my:<bot>" (student/my_opponents.py)
STAGES = [
    "random",
    "basic",
    {"basic": 0.4, "aggressive": 0.3, "defensive": 0.3},
    {"basic": 0.2, "aggressive": 0.3, "defensive": 0.3, "my:camper": 0.2},
]
THRESHOLD = 0.7
WINDOW = 200

# Training conditions (only for training: the tournament always uses the defaults)
ENV_KWARGS = dict(
    maps=None,                   # None = official maps + 50% random maps. Or e.g. ["random"]
    config=None,                 # e.g. {"max_steps": 300}
)

# Algorithm and hyperparameters
ALGO = PPO                       # PPO, A2C, DQN (or sb3_contrib algorithms if installed)
ALGO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,                # per env -> batch of n_steps * N_ENVS transitions
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,               # exploration
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[128, 128]),
)


# ============================================================================
def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths relative to the root
    env = make_vec_env(
        TanksEnv,
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(opponent=STAGES[0], reward_fn=my_reward, **ENV_KWARGS),
        monitor_kwargs=dict(info_keywords=("is_success",)),
    )

    if INIT_MODEL:
        model = ALGO.load(INIT_MODEL, env=env, device="cpu", tensorboard_log="./logs/")
        print(f"Continuing from {INIT_MODEL}")
    else:
        model = ALGO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", device="cpu", seed=SEED,
                     **ALGO_KWARGS)

    callbacks = [
        WinRateCurriculum(STAGES, threshold=THRESHOLD, window=WINDOW),
        CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1), save_path=f"./models/checkpoints_{NAME}/",
                           name_prefix=NAME),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, tb_log_name=NAME,
                reset_num_timesteps=INIT_MODEL is None)
    model.save(f"./models/{NAME}")
    env.close()
    print(f"\nSaved models/{NAME}.zip\n")

    evaluate_agent(f"models/{NAME}.zip", games=10)


if __name__ == "__main__":   # required on Windows for SubprocVecEnv
    main()
