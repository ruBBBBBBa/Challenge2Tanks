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

## 2026-11-06 08:45 · PC1 · league.py

- `student/league.py` pasa de PC3 a PC1 (actualizado CLAUDE.md, secciones 6 y 7).
- Pruebas pequeñas (nice 19, 1 hilo, 1 proceso, 1 mapa): ranking, cara a cara, partidas sospechosas, bracket y replays OK. Modelo contra modelo es determinista: dos ejecuciones dan resultados idénticos.
- Coste medido con A-g1 entrenando en paralelo: unos 300 pasos/s por proceso. Las partidas modelo contra modelo entre checkpoints de A-g1 duran ~466 pasos de media (casi siempre timeout); las de bots, ~55.
- **Aviso:** `models/tanks_ppo.zip` y `models/v1.zip` (antiguos) no cargan con numpy<2 (`No module named 'numpy._core.numeric'`). Es el riesgo de la sección 3, confirmado: no usarlos.

## 2026-11-06 08:31 · PC2 · D-distill (pruebas pequeñas)

- `gen` con 80k pasos y 8 entornos (jue 5, máquina compartida con PC1): **~380 pasos/s**, 305 episodios. Hard determinista: 0.96–1.00 contra basic, aggressive, defensive y bushcamper; 0.44 de victorias y 0.44 de empates contra medium; 100 % de empates contra hard determinista.
- `train` sobre esos 80k (1 proceso, `nice -n 19`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ... train --epochs 2 --seg-len 512 --threads 1`, con A-g1 corriendo): **28–39 s por época, ~2 GB de RSS máximo**. Acuerdo de acciones en validación: 0.31 (KL 1.11). No dice nada todavía: solo hubo 8 pasos de gradiente por época.
- Criterio para D hoy: > 90 % de acuerdo con hard y resultados parecidos contra los bots. Ganar a hard se decide en la liga, después del RL.

## 2026-11-06 ~09:30 · PC1 · map_gen.py + mixed2 + inicios preparados

- `student/map_gen.py`: generador propio simétrico respecto al centro, con estilos open, maze, river (meandros + vados), bushes y classic. Admite densidad y proporciones de muro, ladrillo, agua y arbusto. Spawns como en el del profe; camino entre ellos sin romper ladrillos.
  `python student/map_gen.py --n 400 --seed 0` → 400 mapas en `student/maps/gen/` (80 por estilo), validados con `check_map`.
- `envs.py`: conjuntos `gen` y `mixed2` (35 % official_aug, 30 % random del profe, 35 % gen; proporción comprobada sobre la lista).
- `envs.py`: `scenario_prob`. Con esa probabilidad, después del reset se aplica un inicio preparado: posiciones válidas (Manhattan ≥ 4 y conectadas, aunque sea rompiendo ladrillos), orientaciones, HP 1–3 y cooldown 0–5 aleatorios, y en la mitad de los casos `step_count` entre 250 y 480.
  `damage_dealt` = 3 − HP del rival, para que el desempate por timeout sea coherente. `info["scenario"]` marca esos episodios. `--scenario-prob` en `train_agent.py` (por defecto 0).
- Comprobado: `check_env` OK; en 150 episodios preparados la obs coincide con `vector_obs` (incluidos HP y tiempo), y `result`, `is_success`, `hp`, el desempate y terminated/truncated son correctos. Con prob 0 el entorno es idéntico al normal. Con p = 0.3 salen un 29,6 % de inicios preparados.
  Entrenamiento corto desde hard con `--maps mixed2 --scenario-prob 0.3`: OK.

## 2026-11-06 ~10:00 · PC1 · league.py --gen-maps

- `--gen-maps N` (con `--gen-seed`, por defecto 20261107): añade N mapas de `map_gen.py` generados en la propia liga, uno por cada estilo en rotación y con un RNG independiente por mapa (`default_rng([seed, k])`).
  No repiten ninguno de los 400 de `student/maps/gen/` (comprobado con 50) y son reproducibles.
- Se reportan en una columna aparte, «gen (Elo gen)», con el win rate y un Bradley–Terry propio centrado en la media del Elo principal. Quedan **fuera** del Elo principal, del W-D-L, del top-3, del peor emparejamiento y del bracket. Las partidas sospechosas sí los incluyen.
  Comprobado: el Elo principal es idéntico con y sin `--gen-maps`.
- Coste: cada mapa gen suma lo mismo que un mapa normal. Con los 24 mapas de base, `--gen-maps 16` supone un 67 % más de partidas.

## 2026-11-06 ~10:30 · PC1 · export_tb.py, comportamiento en league.py, datos de imitación

- `student/export_tb.py`: lee `results/tb/<ID>/` y genera `results/curves/<ID>.csv` (una columna por tag), un PNG comparativo por métrica (EMA `--smooth 0.9` sobre la curva cruda) y `<ID>_pfsp_wr.png` si hay `pfsp/wr/*`. Probado con copias de `logs/A-g1_1`.
  **Ojo:** SB3 2.1.0 no registra `rollout/success_rate` en PPO (solo en los algoritmos off-policy), así que **A-g1 no lo tiene**. Desde ahora lo registra `PFSPCallback` (win rate de los últimos 100 episodios), y A-g2 ya lo tendrá. Para A-g1 se puede usar `pfsp/wr/*`.
- `league.py`: tabla «Comportamiento» por participante (mapas principales, sin gen): disparos por partida, puntería (impactos/disparos), balas anuladas, % de victorias por destrucción y por timeout, duración media, empates y % de pasos escondido (en un arbusto e invisible para el rival).
  Cada partida se rehace a partir de sus acciones (`replay_events`) y se comprueba que el ganador y los disparos coinciden con la partida original.
- `league.py --save-all-replays`: guarda todas las partidas en `<salida>_replays.jsonl.gz`, en formato replay más los metadatos de la liga (p0, p1, mapa, lado, semilla). Ocupa ~0,4 KB por partida, unos 6 MB para una liga como la 1. No hace falta subirlo.
- `student/winners_dataset.py`: genera secuencias (obs, acción) del lado ganador, solo de las partidas ganadas por los N mejores modelos del ranking. La obs es `vector_obs`, la recompensa es `default_reward` desde el punto de vista del ganador y la acción va como etiqueta one-hot en `probs`.
  `--verify K` comprueba que el modelo ganador reproduce sus acciones a partir de la obs reconstruida: 10 de 10 en la prueba.
  Prueba pequeña de principio a fin: liga → dataset → `distill.py train` (1 época): OK, **sin modificar distill.py**.

### Notas para PC2 (D-g2): formato y `distill.py train`
- El formato es el mismo que el de `distill.py gen` (`obs`, `probs`, `rew`, `done` con forma (L, E, ·)). Los episodios van seguidos en E streams; la cola de cada stream es relleno con `done=False`, y `returns_and_mask` ya lo enmascara para la pérdida de valor.
- La KL del actor **no** está enmascarada. En el relleno `p=0`, así que no aporta gradiente, pero cuenta en la media (la diluye en 1 − fill; el fill es ~99,8 %). `acc` también cuenta las filas de relleno. Es un efecto pequeño y no hace falta cambiar nada.
- Con etiquetas one-hot, la «kl» que se muestra es −log q(a), es decir, entropía cruzada. No es comparable con la KL de la destilación desde hard.
- **Sesgo del crítico:** todos los episodios son victorias, así que la red de valor aprendería retornos demasiado optimistas. Opciones: `--vf-coef` bajo (0.1) o mezclar con los datos de hard. Para concatenar a lo largo del tiempo (`np.concatenate(axis=0)`), genera el dataset con `--streams` igual al `n_envs` de los datos de hard (8 por defecto).
- Muchas de las victorias son contra bots. Si se quieren solo partidas contra modelos o baselines, habría que añadir un filtro (son pocas líneas).

## 2026-11-06 ~11:30 · PC1 · Análisis de las partidas sospechosas de la liga 1

- En la liga 1 hay 573 partidas que acaban en timeout con 0 daño, 368 de ellas entre modelos y baselines (**38 % de esas 960 partidas**).
  **A-g1_12000k contra A-g1_16000k: 56 de 64 partidas (88 %).**
- Muestra de 40 rehechas con `play_match` (todas reproducen el timeout con 0 daño): 24 de 12000k contra 16000k y 16 de otros emparejamientos de modelos y baselines. Clasificación a partir de las características de cada paso:

| tipo | total | 12k-16k | resto | ejemplo |
|---|---|---|---|---|
| (e) bloqueo diagonal (nuevo) | 30 % | 9/24 | 3/16 | 12000k contra 16000k en arena_02_fortress: diagonal adyacentes, los dos apuntando a la misma celda libre, quietos 400+ pasos |
| (d) bucle: giros, vaivén o empujar un muro | 32,5 % | 9/24 | 4/16 | 16000k contra 12000k en gen_5: uno hace forward/backward y el otro gira a izquierda y derecha, junto a él, sin disparar nunca |
| (a) enfrentados sin dañarse | 17,5 % | 6/24 | 1/16 | 12000k contra 16000k en river_rot90: cara a cara, disparan a la vez, 94 balas anuladas |
| (b) nunca se acercan | 17,5 % | 0/24 | 7/16 | A-g1_4000k contra hard en gen_4: vaivén forward/backward, a distancia ≥ 12 |
| (c) uno se esconde en un arbusto | 2,5 % | 0/24 | 1/16 | A-g1_4000k contra hard en gen_7: hard, en el arbusto el 73 % del tiempo |

- **Bloqueo diagonal:** si los dos avanzan hacia la misma celda, el motor no mueve a ninguno (`_move_tanks`), y en diagonal ninguno tiene línea de tiro. Hay variantes empujando (~450 choques) y quietas (noop). Muchos casos de (d) son el mismo encuentro diagonal adyacente, pero con un baile de giros y vaivén.
- **Lo que se aprende:** entre copias de A (y entre hard y medium) la política simétrica y determinista se bloquea consigo misma. Es probable que pase lo mismo con modelos de otros equipos que también derivan de hard.
  Ideas:
  - inicios preparados en diagonal adyacente (`--scenario-prob`);
  - un bot o un explotador que rompa la simetría (dar un paso atrás o a un lado para alinearse y disparar);
  - castigar más los choques repetidos (`bumped`) o el empate sin daño en las recompensas de PC3.
- GIFs en `out/sospechosas/` (sin git): `a_03`, `a_29`, `b_25`, `b_40`, `c_26`, `d_04`, `d_38`, `e_15` y `e_24`.

## 2026-11-06 ~12:15 · PC1 · Inicios preparados de combate cercano (`--close-prob`)

- `--close-prob` (por defecto 0.5): parte de los inicios preparados (`--scenario-prob`) que son de combate cercano, repartidos 50/50 entre:
  - `diag`: diagonal adyacente. En el 30 % de los intentos se busca que alguna casilla común esté bloqueada (agua, muro, ladrillo). Si hay casilla común libre, en la mitad de los casos los dos tanques miran hacia ella (el bloqueo de la liga 1); en el resto, la orientación es aleatoria.
  - `faceoff`: alineados a 1–5 casillas, con línea de tiro despejada, mirándose de frente y viéndose (a ≥ 3 casillas ninguno está en un arbusto). Cooldowns iguales en la mitad de los casos (como tras anular las balas) y distintos en la otra mitad.
  - El resto de inicios preparados sigue igual (`random`). HP aleatorios y la mitad con `step_count` avanzado, en todos los tipos.
- `info["scenario"]` vale `False` o el tipo de inicio: `"random"`, `"diag"` o `"faceoff"`.
- Comprobado en 600 inicios con close_prob = 1:
  - 46 % diag y 54 % faceoff;
  - diag: 28 % con casilla común bloqueada y 54 % mirando a la casilla libre;
  - faceoff: 47 % con cooldowns iguales; en la obs, «enemigo en mi mira» y «yo en su mira» = 1;
  - 49 % con `step_count` avanzado; obs = `vector_obs`; los finales son correctos;
  - el bloqueo se reproduce: con los dos avanzando hacia la casilla común, ninguno se mueve.
  Con `--scenario-prob 0.3` y la close-prob por defecto: 15,5 % random, 7,2 % diag y 7,7 % faceoff. Entrenamiento corto: OK.

## 2026-11-06 ~11:40 · PC1 · student/pipeline.py (encadenador de generaciones)

- Bucle de `--generations N` (por defecto 2):
  1. espera al PID de `--wait-pid`;
  2. copia las curvas a `results/tb/<ID>/`;
  3. monta `models/ligaK/` con:
     - snapshots de la generación cada 4M pasos, más el último (si el modelo final coincide con el último snapshot, entra solo uno);
     - los 3 mejores modelos de la última liga;
     - los snapshots `B-*`, `C-*`, `D-*` y `X1-*` de `models/league/`;
     - nunca nada de `models/reserved/`;
  4. juega la liga (`--gen-maps 8 --save-all-replays --bot-seeds 1`, 6 bots);
  5. elige el modelo de la rama A con mayor Elo (desempate por P(campeón));
  6. lanza la siguiente generación desacoplada (`--pfsp` variance y después hard, `mixed2`, `--scenario-prob 0.3`, 30M pasos, `ent_coef` 0.02 y la mezcla de rivales nueva).
- Seguridad:
  - no lanza nada si hay otro `train_agent.py` corriendo (lo deja apuntado en el log);
  - `STOP_PIPELINE` en la raíz lo para tras el paso en curso;
  - no sobrescribe un `models/ligaK/` que ya tenga contenido;
  - no hace operaciones de git.
- Trazabilidad: `logs/pipeline.log`, `results/pipeline_ligaK.md` y una entrada en esta bitácora por generación.
- Pruebas:
  - `--dry-run` en el repo real: no escribe nada y detecta A-g2 en marcha («NOT LAUNCHED»).
  - Ejecución mínima en un worktree aislado: A-g2 simulado con 2 snapshots, liga de 1 mapa y entrenamientos de 20k pasos (nice 19), con 2 generaciones encadenadas (liga2 → A-g3 → liga3 → A-g4) en 3 min. El desempate por P(campeón) se probó con un empate real de Elo. `STOP_PIPELINE`: OK.
  - **Fallo encontrado y corregido:** el entrenamiento lanzado por el propio pipeline se quedaba zombi y `kill(pid, 0)` lo daba por vivo. Ahora se recoge con `waitpid(WNOHANG)`.
