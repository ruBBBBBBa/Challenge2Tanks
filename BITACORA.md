# BITÁCORA

## 2026-11-05 17:55 · PC1 · entorno

- CPU: i7-1355U, 10 núcleos (2P+8E), 12 hilos.
- Venv `.venv/` (uv): Python 3.10, numpy 1.26.4, gymnasium 0.29.1, SB3 2.1.0, sb3-contrib 2.1.0, torch 2.2.2+cpu.
  El venv antiguo (Python 3.14, numpy 2.5) **no sirve**: rompe `numpy<2`.
- `tools/checkenv.py`: 7100 pasos/s (random contra random), unos 2500 pasos/s contra los bots heurísticos (un solo proceso).
- Entrenamiento real desde hard con la mezcla de la generación 1 (rival hard en el pool), 10 entornos: **unos 380 pasos/s** (con pruebas en paralelo).
- Hard determinista contra `my:stoch:baseline:hard`: 2 victorias, 4 empates y 6 derrotas en 12 partidas. El hard estocástico es un rival útil.
- Bots propios (12 partidas): rusher 92 % contra basic y 100 % contra defensive; bushcamper 92 % y 58 %; hitandhide 67 % y 83 %. Los tres ganan el 0 % contra hard.

## 2026-11-06 ~08:00 · PC1 · A-g1 (lanzado)

- Comando: `python student/train_agent.py --name A-g1 --init baseline:hard --reward default --opponents '{"baseline:hard":0.25,"my:stoch:baseline:hard":0.20,"baseline:medium":0.10,"basic":0.15,"aggressive":0.15,"defensive":0.15}' --timesteps 40000000 --n-envs 10 --lr 1e-4 --clip-range 0.15 --ent-coef 0.01 --maps mixed --snapshot-every 2000000 --eval-games 2`
- Salida en `logs/A-g1.out` y PID en `logs/A-g1.pid`. LR efectivo 1e-4, comprobado.
- Nota: a unos 380 pasos/s, 40M pasos son ~29 h. Snapshots cada 2M (~1,5 h) en `models/league/A-g1_*k.zip`.
- Smoke tests de `train_agent.py` (de hard a fin, liga + PFSP con `every=3`): OK.
