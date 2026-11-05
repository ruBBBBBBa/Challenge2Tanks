# Baselines

Teacher-trained PPO models used as opponents and as reference:

| File         | Spec              |
|--------------|-------------------|
| `easy.zip`   | `baseline:easy`   |
| `medium.zip` | `baseline:medium` |
| `hard.zip`   | `baseline:hard`   |

```python
env = TanksEnv(opponent="baseline:easy")
```

If this folder is empty, your teacher has not published the baselines yet.
