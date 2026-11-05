# Final submissions

```
submissions/<your_name>/model.zip
```

- Only the `model.zip` (a Stable-Baselines3 model) is used in the tournament.
- It must be trained on the **official observation** (`Box(77,)`, `observations.vector_obs`)
  and the official action space (`Discrete(6)`). It is loaded automatically
  (PPO, A2C, DQN, or sb3-contrib algorithms if installed).
- Check it before submitting:

```bash
python tools/evaluate.py --agent submissions/<your_name>/model.zip
```

The report is delivered separately (see README.md at the root).
