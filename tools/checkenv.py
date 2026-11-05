# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""Sanity checks for the environment: Gymnasium/SB3 API, a random rollout and a speed test."""
import os
import sys
import time

from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.tanks_env import TanksEnv  # noqa: E402

if __name__ == "__main__":
    env = TanksEnv(opponent="basic")
    check_env(env, warn=True)
    print("check_env: OK")

    obs, info = env.reset(seed=0)
    print("observation shape:", obs.shape, "| first info:", info)

    for opponent in ["random", "basic", "aggressive", "defensive"]:
        env = TanksEnv(opponent=opponent)
        env.reset(seed=1)
        steps, episodes, t0, results = 0, 0, time.perf_counter(), []
        while episodes < 10:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                results.append(info["result"])
                episodes += 1
                env.reset()
        dt = time.perf_counter() - t0
        print(f"random learner vs {opponent:10s}: {steps / dt:7.0f} steps/s | "
              f"results {dict((r, results.count(r)) for r in set(results))}")
