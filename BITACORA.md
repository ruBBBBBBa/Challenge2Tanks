
## 2026-11-05 18:06 · PC1 · smoke

- Comando: `python student/train_agent.py --name smoke --timesteps 8192 --n-envs 4 --snapshot-every 4096 --eval-games 2`
- Pasos: 8,192 · inicio 2026-11-05T18:02 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 0 | 2 | 0 | 0% | 0.00 |
| baseline:hard | 0 | 2 | 0 | 0% | 0.00 |
| baseline:medium | 0 | 2 | 0 | 0% | 0.00 |

- Notas: 
