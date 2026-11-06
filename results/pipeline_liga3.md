# Pipeline · liga3 · 2026-11-06 17:03:22

Generación terminada: **A-g3**. Siguiente: **A-g4**.

## Participantes (modelos)

- `A-g3_4000k.zip`: A-g3 cada 4M
- `A-g3_8000k.zip`: A-g3 cada 4M
- `A-g3_12000k.zip`: A-g3 cada 4M
- `A-g3_16000k.zip`: A-g3 cada 4M
- `A-g3_20000k.zip`: A-g3 cada 4M
- `A-g3_24000k.zip`: A-g3 cada 4M
- `A-g3_28000k.zip`: A-g3 cada 4M
- `A-g3.zip`: A-g3 último
- `A-g2_20000k.zip`: top-3 de league_2026-11-06_1335_liga2.json (Elo 2168)
- `A-g2_8000k.zip`: top-3 de league_2026-11-06_1335_liga2.json (Elo 2154)
- `A-g2.zip`: top-3 de league_2026-11-06_1335_liga2.json (Elo 2150)
- `baseline:hard`, `baseline:medium` y los bots: `basic`, `aggressive`, `defensive`, `my:rusher`, `my:bushcamper`, `my:hitandhide`

## Liga

Resultados: `results/league_2026-11-06_1659_liga3.json` (y `.md`).

`/home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/league.py --models 'models/liga3/*.zip' --baselines hard medium --bots basic aggressive defensive my:rusher my:bushcamper my:hitandhide --bot-seeds 1 --gen-maps 8 --save-all-replays --procs 10 --tag liga3`

| # | participante | Elo | win rate | vs top-3 | aleatorios | gen | P(campeón) |
|---|---|---|---|---|---|---|---|
| 1 | A-g3 | 2350 | 78% | 38% | 76% | 69% | 23% |
| 2 | A-g3_20000k | 2338 | 76% | 29% | 74% | 66% | 7% |
| 3 | A-g3_28000k | 2333 | 75% | 31% | 74% | 65% | 13% |
| 4 | A-g3_24000k | 2277 | 71% | 21% | 72% | 73% | 14% |
| 5 | A-g3_16000k | 2258 | 71% | 22% | 69% | 73% | 18% |
| 6 | A-g3_8000k | 2255 | 70% | 10% | 70% | 64% | 3% |
| 7 | A-g3_12000k | 2221 | 67% | 12% | 73% | 66% | 3% |
| 8 | A-g3_4000k | 2211 | 66% | 12% | 67% | 59% | 18% |
| 9 | A-g2_8000k | 1791 | 48% | 0% | 49% | 44% | - |
| 10 | A-g2_20000k | 1790 | 48% | 0% | 48% | 46% | - |
| 11 | A-g2 | 1784 | 48% | 0% | 48% | 47% | - |
| 12 | baseline:hard | 1500 | 35% | 0% | 33% | 34% | - |

## Elegido

`models/liga3/A-g3.zip`: mayor Elo principal de la rama A (2350).

## Lanzamiento

Lanzado (PID 769646).

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g4 --init models/liga3/A-g3.zip --league-glob 'models/liga3/*.zip' --league-share 0.5 --pfsp hard --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g4.out 2>&1 &
```
