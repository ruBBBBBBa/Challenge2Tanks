# Pipeline · liga2 · 2026-11-06 13:38:31

Generación terminada: **A-g2**. Siguiente: **A-g3**.

## Participantes (modelos)

- `A-g2_4000k.zip`: A-g2 cada 4M
- `A-g2_8000k.zip`: A-g2 cada 4M
- `A-g2_12000k.zip`: A-g2 cada 4M
- `A-g2_16000k.zip`: A-g2 cada 4M
- `A-g2_20000k.zip`: A-g2 cada 4M
- `A-g2.zip`: A-g2 último
- `A-g1_16000k.zip`: top-3 de league_2026-11-06_1046_liga1b.json (Elo 1836)
- `A-g1_12000k.zip`: top-3 de league_2026-11-06_1046_liga1b.json (Elo 1762)
- `A-g1_8000k.zip`: top-3 de league_2026-11-06_1046_liga1b.json (Elo 1683)
- `baseline:hard`, `baseline:medium` y los bots: `basic`, `aggressive`, `defensive`, `my:rusher`, `my:bushcamper`, `my:hitandhide`

## Liga

Resultados: `results/league_2026-11-06_1335_liga2.json` (y `.md`).

`/home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/league.py --models 'models/liga2/*.zip' --baselines hard medium --bots basic aggressive defensive my:rusher my:bushcamper my:hitandhide --bot-seeds 1 --gen-maps 8 --save-all-replays --procs 10 --tag liga2`

| # | participante | Elo | win rate | vs top-3 | aleatorios | gen | P(campeón) |
|---|---|---|---|---|---|---|---|
| 1 | A-g2_20000k | 2168 | 80% | 43% | 84% | 76% | 14% |
| 2 | A-g2_8000k | 2154 | 80% | 38% | 80% | 72% | 16% |
| 3 | A-g2 | 2150 | 79% | 38% | 80% | 78% | 37% |
| 4 | A-g2_4000k | 2126 | 80% | 38% | 77% | 66% | 17% |
| 5 | A-g2_16000k | 2091 | 75% | 18% | 73% | 71% | 11% |
| 6 | A-g2_12000k | 2085 | 74% | 24% | 69% | 71% | 5% |
| 7 | A-g1_16000k | 1742 | 52% | 1% | 50% | 52% | 0% |
| 8 | A-g1_12000k | 1698 | 47% | 0% | 47% | 45% | 0% |
| 9 | A-g1_8000k | 1644 | 46% | 0% | 47% | 43% | - |
| 10 | baseline:hard | 1500 | 40% | 0% | 38% | 41% | - |
| 11 | baseline:medium | 1466 | 38% | 0% | 38% | 37% | - |
| 12 | my:bushcamper | 1202 | 23% | 0% | 19% | 19% | - |

## Elegido

`models/liga2/A-g2_20000k.zip`: mayor Elo principal de la rama A (2168).

## Lanzamiento

Lanzado (PID 665930).

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g3 --init models/liga2/A-g2_20000k.zip --league-glob 'models/liga2/*.zip' --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g3.out 2>&1 &
```
