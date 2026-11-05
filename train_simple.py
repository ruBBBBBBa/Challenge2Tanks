"""
Simplest possible training: create the environment and train a default PPO against "basic".

    python train_simple.py

Next steps: train.py (parallel envs + your reward) and student/train_advanced.py (everything).
"""
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from tanks import TanksEnv

# Configuration
total_timesteps = 100_000
env = TanksEnv(opponent="basic", render_mode='human')

# TODO: Explore PPO hyperparameters (learning_rate, n_steps, batch_size, gamma, etc.)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# Train the agent
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("./models/tanks_simple")

# Evaluate the agent (mean reward + win rate against "basic")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

wins = 0
for episode in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    wins += info["result"] == "win"
print(f"Win rate vs basic: {wins}/10")

env.close()
