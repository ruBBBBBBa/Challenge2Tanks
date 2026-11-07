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

## 2026-11-06 13:04 · PC3 · T-agr

- Comando: `python student/train_agent.py --name T-agr --reward agresiva --init baseline:hard --timesteps 50000 --lr 1e-4 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --n-envs 2 --blend-steps 20000 --snapshot-every 10000000 --eval-games 4 --opponents {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10, "basic": 0.10, "aggressive": 0.10, "defensive": 0.10, "my:rusher": 0.05, "my:bushcamper": 0.05, "my:hitandhide": 0.05}`
- Pasos: 57,344 · inicio 2026-11-06T13:03 · lr efectivo 0.0001
- Evaluación determinista (4 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 3 | 1 | 0 | 75% | 2.25 |
| basic | 4 | 0 | 0 | 100% | 3.00 |
| aggressive | 4 | 0 | 0 | 100% | 3.00 |
| defensive | 4 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 2 | 0 | 50% | 1.50 |
| baseline:hard | 0 | 4 | 0 | 0% | 0.00 |
| baseline:medium | 4 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 13:04 · PC3 · T-sup

- Comando: `python student/train_agent.py --name T-sup --reward superviviente --init baseline:hard --timesteps 50000 --lr 1e-4 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --n-envs 2 --blend-steps 20000 --snapshot-every 10000000 --eval-games 4 --opponents {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10, "basic": 0.10, "aggressive": 0.10, "defensive": 0.10, "my:rusher": 0.05, "my:bushcamper": 0.05, "my:hitandhide": 0.05}`
- Pasos: 57,344 · inicio 2026-11-06T13:03 · lr efectivo 0.0001
- Evaluación determinista (4 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 2 | 0 | 50% | 1.50 |
| basic | 4 | 0 | 0 | 100% | 3.00 |
| aggressive | 4 | 0 | 0 | 100% | 3.00 |
| defensive | 4 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 2 | 0 | 50% | 1.50 |
| baseline:hard | 0 | 4 | 0 | 0% | 0.00 |
| baseline:medium | 2 | 0 | 2 | 50% | 2.50 |

- Notas: 

## 2026-11-06 13:04 · PC3 · T-gen

- Comando: `python student/train_agent.py --name T-gen --reward generalista --init baseline:hard --timesteps 50000 --lr 1e-4 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --n-envs 2 --blend-steps 20000 --snapshot-every 10000000 --eval-games 4 --opponents {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10, "basic": 0.10, "aggressive": 0.10, "defensive": 0.10, "my:rusher": 0.05, "my:bushcamper": 0.05, "my:hitandhide": 0.05}`
- Pasos: 57,344 · inicio 2026-11-06T13:03 · lr efectivo 0.0001
- Evaluación determinista (4 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 2 | 0 | 50% | 1.50 |
| basic | 4 | 0 | 0 | 100% | 3.00 |
| aggressive | 4 | 0 | 0 | 100% | 3.00 |
| defensive | 3 | 1 | 0 | 75% | 2.25 |
| baseline:easy | 2 | 2 | 0 | 50% | 1.50 |
| baseline:hard | 0 | 4 | 0 | 0% | 0.00 |
| baseline:medium | 4 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 ~13:30 · PC3 · my_rewards.py, pruebas de 50k y lanzamiento de B-g1 y C-g1

- PC3: Windows sin WSL, Python 3.10, 6 núcleos físicos / 12 lógicos. `checkenv`: OK. numpy 1.26.4, SB3 y sb3-contrib 2.9.2.
- `student/my_rewards.py`: clases `GeneralistaReward`, `AgresivaReward` y `SupervivienteReward` (subclases de `ShapedReward`) con `blend` y `shaping_scale`. Shaping por potencial F = 0.995·Φ(s′) − Φ(s), con Φ = 0 en el estado terminal. Términos de Φ: distancia normalizada (si el enemigo no se ve, la máxima), enemigo en mi mira con el arma cargada, el mismo con el enemigo recargando (agresiva), bala enemiga que viene hacia mí a ≤ 6 casillas y yo en su mira con cooldown > 1 (superviviente).
  - Agresiva: hits_dealt 3, hits_taken −1.5, draw −5, step −0.01.
  - Superviviente: hits_taken −3.5, step −0.003.
- Pruebas de 50k pasos desde hard (T-gen, T-agr, T-sup; 2 envs cada una, ~1080 fps): sin errores, lr efectivo 1e-4.
- Revisión de 25 partidas deterministas por modelo (4 mapas oficiales + 1 aleatorio × hard, rusher, bushcamper, hitandhide y basic):
  - apuntar con el arma cargada sin disparar: 0 casos en todos los modelos;
  - esconderse en un arbusto al final: 0 %;
  - empates sin daño: solo contra hard (espejo determinista, igual que hard contra sí mismo);
  - rachas largas de giros: ya están en hard (180 seguidos contra bushcamper). En T-agr y T-sup hay una de ~450 en un empate contra hard en un mapa aleatorio. **Vigilarlo en la liga 2.**
- B-g1 (agresiva) y C-g1 (superviviente) lanzados con Start-Process y prioridad BelowNormal, `--n-envs 3` cada uno, OMP/MKL = 1, con los argumentos del encargo. ~1270 fps cada uno (≈ 4,5 h para 20M).
## 2026-11-06 13:35 · PC1 · A-g2

- Comando: `python student/train_agent.py --name A-g2 --init models/liga1/A-g1_16000k.zip --reward default --opponents {"baseline:hard":0.20,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15} --league-glob models/liga1/*.zip --league-share 0.5 --pfsp variance --timesteps 20000000 --n-envs 10 --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --snapshot-every 2000000 --eval-games 2`
- Pasos: 20,008,170 · inicio 2026-11-06T10:54 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 3.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 13:38 · PC1 · pipeline: A-g2 → liga2 → A-g3

- Liga: `results/league_2026-11-06_1335_liga2.json` (9 modelos + baselines + bots).
- Elegido: `models/liga2/A-g2_20000k.zip` (mayor Elo principal de la rama A (2168)).
- Lanzado A-g3 (PID 665930): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g3 --init models/liga2/A-g2_20000k.zip --league-glob 'models/liga2/*.zip' --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g3.out 2>&1 &`
- Resumen: `results/pipeline_liga2.md`.

## 2026-11-06 14:30 · PC2 (torre, Windows, 8 núcleos físicos / 16 hilos) · D-distill y D-g1

- `gen --steps 1500000 --n-envs 8`: ~4 min (~6000 pasos/s), 5672 episodios. Hard determinista: 1.00 contra basic y aggressive, 0.89 contra defensive, 0.42 de victorias (0.44 de empates) contra medium y 1.00 de empates contra hard.
- `train --epochs 20 --threads 8` (con seg-len 2048 por defecto, 728 segmentos): ~55–60 s por época, 20 min en total. **Acuerdo en validación: 0.906** (KL 0.061). Ha ido subiendo: 0.53 en la época 1, 0.80 en la 7 y 0.88 en la 12.
- `evaluate.py --games 40`, determinista (D-distill / hard):
  - basic: 97.5 / 100;
  - aggressive: 100 / 100;
  - defensive: 92.5 / 97.5;
  - random: 60 / 52.5;
  - easy: 5 / 25;
  - medium: 27.5 / 65;
  - hard: 0 / 0 (38 y 40 empates).
  Contra los bots juega parecido a hard, pero rinde peor contra los baselines.
- Lanzado D-g1 (PID 11512), con la mezcla de rivales del plan pero quitando medium para meter `model:models/league/A-g2_20000k.zip` (0.15): `train_agent.py --name D-g1 --algo recurrent --init models/D-distill.zip --lr 5e-5 --timesteps 60000000 --snapshot-every 2000000 --maps mixed2 --scenario-prob 0.3 --ent-coef 0.02 --eval-games 2 --n-envs 8`.
  - Script: `logs/run_D-g1.ps1`; salida en `logs/D-g1.out` y `logs/D-g1.err`.
  - LR efectivo de 5e-5, comprobado. **~200 pasos/s**: domina la actualización de RecurrentPPO en 1 hilo. Ritmo de ~0,7M por hora y ~2,8 h por snapshot.

## 2026-11-06 ~15:15 · PC2 · `--learner-threads` en train_agent.py y relanzamiento de D-g1

- **Cambio en `student/train_agent.py` (archivo de PC1, mínimo):** nueva opción `--learner-threads N` (por defecto 1, sin cambios respecto a lo que había).
  Con N > 1 se añade `LearnerThreadsCallback`, que solo actúa en el proceso principal:
  - `_on_rollout_end` hace `torch.set_num_threads(N)` para la actualización;
  - `_on_rollout_start` vuelve a 1.
  Los entornos siguen con 1 hilo.
- **Medida** (RecurrentPPO, la config de D-g1 y 5 iteraciones, con el D-g1 anterior corriendo a la vez):
  - 1 hilo: ~57 s por iteración de 8192 pasos (~145 pasos/s);
  - 8 hilos: ~41 s (~200 pasos/s).
  Relanzado sin competencia de CPU: ~35 s por iteración, **~240 pasos/s marginales** (~0,85M por hora) frente a ~150–200 antes.
  Útil sobre todo con `--algo recurrent`, donde domina la actualización.
- **D-g1:** se paró el primero (PID 11512, 1,1M pasos; curvas en `logs/D-g1_1` y salida en `logs/D-g1.run1.out`) y se relanzó desde `models/D-distill.zip` con la misma config más `--learner-threads 8`.
  PID 24672 (lanzador; el proceso real es el 23772). Las curvas nuevas van a `logs/D-g1_2/`.

## 2026-11-06 16:48 · PC3 · C-g1

- Comando: `python student/train_agent.py --name C-g1 --reward superviviente --init baseline:hard --timesteps 20000000 --lr 1e-4 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --snapshot-every 2000000 --eval-games 2 --n-envs 3 --opponents {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10, "basic": 0.10, "aggressive": 0.10, "defensive": 0.10, "my:rusher": 0.05, "my:bushcamper": 0.05, "my:hitandhide": 0.05}`
- Pasos: 20,008,170 · inicio 2026-11-06T13:06 · lr efectivo 0.0001
## 2026-11-06 16:58 · PC1 · A-g3

- Comando: `python student/train_agent.py --name A-g3 --init models/liga2/A-g2_20000k.zip --league-glob models/liga2/*.zip --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents {"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}`
- Pasos: 30,008,160 · inicio 2026-11-06T13:38 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 3.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 16:48 · PC3 · B-g1

- Comando: `python student/train_agent.py --name B-g1 --reward agresiva --init baseline:hard --timesteps 20000000 --lr 1e-4 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --snapshot-every 2000000 --eval-games 2 --n-envs 3 --opponents {"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10, "basic": 0.10, "aggressive": 0.10, "defensive": 0.10, "my:rusher": 0.05, "my:bushcamper": 0.05, "my:hitandhide": 0.05}`
- Pasos: 20,008,170 · inicio 2026-11-06T13:06 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 3.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 1.00 |

- Notas: 

## 2026-11-06 ~17:15 · PC3 · Evaluación de B-g1 y C-g1 y snapshots para la liga 2

- Terminados a las 16:48: 20M pasos cada uno, sin errores, ~1500 fps cada uno. Tasa de éxito en el entrenamiento: 0,96 en los dos.
- `tools/evaluate.py --games 20` (determinista):

| rival | hard | B-g1 | C-g1 |
|---|---|---|---|
| random | 55 % | 100 % | 100 % |
| basic | 100 % | 100 % | 100 % |
| aggressive | 100 % | 100 % | 100 % |
| defensive | 95 % | 100 % | 90 % |
| baseline:easy | 20 % | 95 % | 75 % |
| baseline:hard | 0 % (20 empates) | 95 % | 100 % |
| baseline:medium | 70 % | 100 % | 80 % |
| **total** | 62,9 % | **98,6 %** | **92,1 %** |

- Cara a cara en 8 mapas (4 oficiales + 4 aleatorios) × 2 lados:
  - B-g1 contra hard: 16-0-0, daño 48-6;
  - C-g1 contra hard: 16-0-0, daño 48-0;
  - **B-g1 contra C-g1: 16 empates, daño 2-2.** Se bloquean entre ellos, como hard contra sí mismo. Es la debilidad a vigilar en la liga 2.
- Trampas (25 partidas cada uno): las rachas largas de giros heredadas de hard han desaparecido (máximo 3-6 seguidos frente a 180 de hard). 0 empates sin daño contra bots y nada de esconderse en arbustos al final. Episodios mucho más cortos (58-73 pasos de media frente a 199 de hard). Apuntar con el arma cargada sin disparar: 6/63 (B) y 2/64 (C) pasos; conviene revisarlo en replays.
- Snapshots subidos para la liga 2: `B-g1_10000k`, `B-g1_20000k`, `C-g1_10000k`, `C-g1_20000k`. Curvas en `results/tb/B-g1` y `results/tb/C-g1`.
## 2026-11-06 17:03 · PC1 · pipeline: A-g3 → liga3 → A-g4

- Liga: `results/league_2026-11-06_1659_liga3.json` (11 modelos + baselines + bots).
- Elegido: `models/liga3/A-g3.zip` (mayor Elo principal de la rama A (2350)).
- Lanzado A-g4 (PID 769646): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g4 --init models/liga3/A-g3.zip --league-glob 'models/liga3/*.zip' --league-share 0.5 --pfsp hard --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g4.out 2>&1 &`
- Resumen: `results/pipeline_liga3.md`.

## 2026-11-06 ~18:05 · PC3 · student/pipeline_exploiter.py (B y C como exploiters de A)

- Nuevo `student/pipeline_exploiter.py --branch B|C`: bucle hasta que exista `STOP_EXPLOITERS`. Cada generación hace:
  1. `sync_models.py --once`;
  2. pool `models/pool_<rama>_gK/` con las 3 A más recientes, la última de la otra rama, D-g* y X1 si existen, y la mejor anterior propia (nunca R1 ni `models/reserved/`);
  3. `train_agent.py` con la recompensa y los hiperparámetros de g1, `--league-share 0.6 --pfsp hard`, mixed2, `--scenario-prob 0.3`, 6M pasos (~75 min a ~1400 fps), un snapshot cada 1,5M y `--n-envs 3`;
  4. evaluación determinista de cada snapshot y del init: 40 partidas contra la A más reciente (4 oficiales + 16 official_aug × 2 lados) y 20 contra hard (4 oficiales + 6 aleatorios × 2 lados). Se publica en `models/outbox/` el mejor contra A que gane ≥ 85 % a hard;
  5. curvas, `results/exploiters_<rama>.md` y una entrada aquí;
  6. `sync_models.py --once --publish B C`.
  El estado queda en `results/exploiters_<rama>.json`, así que se reanuda solo.
- **Cambio en `student/sync_models.py` (archivo de PC1, mínimo):** `fcntl` no existe en Windows. Ahora se usa `fcntl` si está y, si no, `msvcrt.locking` (mismo comportamiento: espera al otro proceso). El mensaje de commit usa `$PC` en lugar de «PC1» fijo. Probado `--once` en Windows: clona `../Challenge2Tanks-sync` y baja 8 modelos.
- **Ojo:** `--publish` recibe las ramas separadas por espacios (`--publish B C`); con `B,C` no publica nada.
- Punto de partida contra `A-g4_4000k` (40 partidas): B-g1_20000k 0-4-36 (5 %) y C-g1_20000k 0-20-20 (25 %).
- Prueba de punta a punta con 32k pasos y sin publicar: OK (artefactos borrados).

## 2026-11-06 ~18:30 · PC2 · GPU para RecurrentPPO: `--device` y benchmark

- GPU: RTX 3080 Ti, 12 GB, driver 591.86 (CUDA 13.1).
  - `.venv` es solo CPU: torch 2.15.0+cpu, SB3 2.9.2 y sb3-contrib 2.9.2.
  - Nuevo entorno `.venv-gpu`, sin git: torch **2.14.1+cu130** (la 2.15 no existe para CUDA), numpy 1.26.4 y el resto de paquetes en las mismas versiones que `.venv`.
- **`student/train_agent.py`:** nueva opción `--device {cpu,cuda,auto}`, **por defecto cpu**. SB3, con "auto", cogería la GPU sola en cuanto hubiera torch con CUDA. Solo se usa en `Algo(...)` y en `Algo.load(...)`; nada más cambia.
- **Benchmark** con la config de D-g1:
  - 5 iteraciones de 8192 pasos; en la media se descarta la primera, que es de calentamiento;
  - con D-g1 (CPU, 8 hilos de learner) corriendo en los dos casos;
  - `collect_rollouts` y `train` se cronometran por separado con un envoltorio (fuera del repo).

| | rollout | actualización | total por iteración | pasos/s |
|---|---|---|---|---|
| CPU, `--learner-threads 8` | 3.5 s | 51.2 s | 54.7 s | ~150 |
| `--device cuda` | 5.2 s | 31.2 s | 36.4 s | ~225 |

  En total, la **GPU es 1,50 veces más rápida**. El rollout es algo más lento en GPU (inferencia de lotes de 8 con la LSTM), pero la actualización es 1,6 veces más rápida.
  **Ojo:** la versión CPU sufre más la competencia con D-g1 (los dos piden 8 hilos en la actualización). D-g1 solo en CPU iba a ~35 s por iteración, así que con la máquina libre la ventaja real de la GPU puede ser menor. Lo seguro: con la GPU la CPU queda libre para otro entrenamiento en paralelo.
- **Zip guardado entrenando en GPU** (`.venv-gpu`, `device=cuda`, `learn(4096)`):
  - carga con `RecurrentPPO.load(..., device="cpu")` en `.venv` (torch sin CUDA);
  - juega con `play_match("model:...")`: gana a basic (3-0, destrucción) y empata con hard (timeout).

## 2026-11-06 ~19:00 · PC2 · D-g1b (D-g1 retomado en GPU)

- Se paró D-g1 (CPU, 1,75M pasos) y se retomó desde `models/checkpoints_D-g1/D-g1_1500000_steps.zip`. Las snapshots de 2M no llegaron a existir.
  `.venv-gpu\Scripts\python.exe student\train_agent.py --name D-g1b --init models/checkpoints_D-g1/D-g1_1500000_steps.zip --device cuda`, con la misma config que D-g1 (`logs/run_D-g1b.ps1`, salida en `logs/D-g1b.out`).
  **Ojo:** el contador de pasos vuelve a 0, así que D-g1b_Nk lleva en realidad 1,5M + N pasos.
- Ritmo sin otro proceso en la máquina: ~33 s por iteración (~250 pasos/s), con la GPU al ~60 %. Es **prácticamente igual** que D-g1 solo en CPU con `--learner-threads 8` (~35 s).
  El 1,5× del benchmark se debía a la competencia por la CPU. La ganancia real es que la CPU queda libre para otro entrenamiento.

## 2026-11-06 ~18:45 · PC2 · Compatibilidad SB3 2.1.0 y lanzamiento de E-g1 (anti-empates)

- **Versiones.** Nuevo `.venv-sb21` (sin git): SB3 2.1.0, sb3-contrib 2.1.0, gymnasium 0.29.1, numpy 1.26.4 y torch 2.15.0+cpu. Con Python 3.12 no hay torch 2.0.x.
  - Prueba con el cargador oficial `ModelAgent(path, deterministic=True)` y 10 partidas contra baseline:hard (4 oficiales, lados alternos).
  - En `.venv-sb21` y en `.venv` (SB3 2.9.2) salen **resultados idénticos**:
    - `D-distill.zip`: 0-10-0;
    - `D-g1_1500000_steps.zip`: 0-10-0;
    - zip guardado entrenando en GPU: 0-8-2.
  - **Los zips guardados con SB3 2.9.2 (CPU o GPU) cargan y juegan igual en 2.1.0.** Falta comprobar torch 2.0.x, que necesita Python ≤ 3.11.
- **`student/my_rewards.py`:** nuevo preset `"nodraw"` = `WeightedReward({"draw": -6.0})`. Es la recompensa por defecto con draw −6; win y lose siguen en ±10. No tiene `blend`, así que se usa tal cual desde el paso 0.
- **sync:** `PC=PC2 sync_models.py --loop --publish D E` (PID 33884, `logs/sync.out`). Ha bajado 8 modelos a `models/shared/`.
- **E-g1** (PID 35892, `logs/run_E-g1.ps1`, `logs/E-g1.out`):
  - no hay ninguna entrada de A-g5 en la bitácora, así que se usa **la config de A-g4** (la última de A apuntada);
  - cambios respecto a A-g4: `--init models/shared/A-g4_4000k.zip --reward nodraw --league-glob 'models/shared/*.zip' --pfsp hard --timesteps 12000000 --snapshot-every 2000000 --n-envs 6`;
  - el resto, igual que A-g4: `--league-share 0.5 --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --eval-games 2` y la mezcla de rivales de A-g4.
  - Liga: los 8 de `models/shared/`. **~2100 pasos/s**, así que los 12M tardan ~1,6 h.
  - D-g1b, en GPU, pasa de ~33 a ~34–36 s por iteración con E-g1 corriendo (~5 % más lento).

## 2026-11-06 19:20 · PC3 · C-g2

- Comando: `python student/train_agent.py --name C-g2 --init models/league/C-g1_20000k.zip --reward superviviente --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_C_g2/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T18:00 · lr efectivo 0.0001
## 2026-11-06 17:34 · PC1 · pipeline: A-g4 → liga4 → A-g5

- Liga: `results/league_2026-11-06_1732_liga4.json` (8 modelos + baselines + bots).
- Elegido: `models/liga4/A-g4_4000k.zip` (mayor Elo principal de la rama A (2409)).
- Lanzado A-g5 (PID 787084): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g5 --init models/liga4/A-g4_4000k.zip --league-glob 'models/liga4/*.zip' --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g5.out 2>&1 &`
- Resumen: `results/pipeline_liga4.md`.

## 2026-11-06 17:55 · PC1 · Liga entre ordenadores: sync_models.py + pipeline

- `student/sync_models.py` (commit `5b49eb5`): todo el git va en un clon aparte, `../Challenge2Tanks-sync`; nunca en la carpeta de trabajo.
  - Bajar: `pull --rebase` y los `.zip` nuevos o cambiados de `models/league/` pasan a `models/shared/`.
  - Subir: `models/outbox/<P>-*.zip` → `git add` solo de ese archivo, commit y push (3 reintentos con pull --rebase) → `models/outbox/sent/`.
  - Nunca publica R1, nada de `models/reserved/` (aunque llegue por enlace simbólico) ni archivos de más de 50 MB.
  - Un lock evita que el `--loop` y el `--once` del pipeline se pisen. Log en `logs/sync.log`.
  - `models/shared/` y `models/outbox/` van en `.gitignore`.
- Prueba de `--once`: clonó el repo y bajó 7 modelos (A-g1_16000k, A-g2_20000k, A-g4_4000k, B-g1_10000k/20000k y C-g1_10000k/20000k). La segunda ejecución no copia nada. Filtros de publicación comprobados.
- `pipeline.py`:
  - ejecuta `sync_models.py --once` antes de montar cada liga;
  - de las otras ramas (B, C, D, X1) coge las **2 versiones más recientes** de cada una, juntando `models/league/` y `models/shared/` (D entra porque sb3-contrib 2.1.0 carga RecurrentPPO); nunca R1 ni `models/reserved/`;
  - tras cada liga, copia el mejor A a `models/outbox/` con su nombre de snapshot (`A-g3.zip` → `A-g3_30000k.zip`).
- Publicado: `A-g3_30000k.zip` (commit `14324fa` hecho por el sync); `A-g4_4000k.zip` ya estaba en origin (idéntico).
- Sync en marcha: `nohup .venv/bin/python student/sync_models.py --loop --publish A > logs/sync.out 2>&1 &`, **PID 795544** (`logs/sync.pid`), cada 15 min.
- **Ojo con `git pull` en esta carpeta:** cada A publicado existe aquí sin seguimiento en `models/league/`, y git se niega a sobrescribirlo («untracked working tree files would be overwritten»). Si pasa, hay que borrar la copia local de esos archivos (es el mismo modelo) y repetir el pull.
- Pendiente, a la espera de confirmación: relanzar el pipeline con el código nuevo esperando a A-g5 (PID 787084, acaba hacia las 21:00 a ~2 430 fps), con 8 generaciones de 12M pasos (~88 min cada una: 82 de entrenamiento y ~5 de liga; terminarían hacia las 8:45).

## 2026-11-06 18:00 · PC1 · Pipeline 3 relanzado + student/safe_pull.sh

- Antes de parar nada se comprobó que A-g5 (PID 787084) seguía vivo. Después se paró **solo** el pipeline viejo (PID 777120); A-g5 siguió en marcha, en su propia sesión.
- Pipeline relanzado con el código nuevo (sync antes de cada liga, otras ramas desde shared y outbox): **PID 798880** (`logs/pipeline.pid`), salida en `logs/pipeline3.out`:
  `nohup .venv/bin/python student/pipeline.py --wait-pid logs/A-g5.pid --generations 8 --timesteps 12000000 --pfsp-modes hard,variance,hard,variance,hard,variance,hard,variance > logs/pipeline3.out 2>&1 &`
  Generaciones A-g6 a A-g13, alternando PFSP hard y variance para comparar los dos modos (A-g5 es variance). A-g5 acabará hacia las 21:00, a ~2 430 fps, y cada generación debería tardar unos 88 min: el final se espera hacia las 8:45.
- `student/safe_pull.sh` (`bash student/safe_pull.sh`):
  1. `git fetch`;
  2. cada archivo que el pull añadiría y que aquí existe sin seguimiento: si su `git hash-object` coincide con el blob de origin, se borra la copia local; si es distinto, se renombra a `<nombre>.local` con aviso;
  3. `git pull --rebase --autostash`;
  4. si el único conflicto es BITACORA.md, conserva las líneas de los dos lados (sin marcas; también quita la sección base del estilo diff3), hace `git add` y descarta el autostash, o continúa el rebase si el conflicto viene de un commit local. Si hay conflicto en otro archivo, para con exit 2 sin tocar nada.
  Nunca usa `git add .`, `git stash` a secas ni `git reset --hard`.
- Pruebas en un sandbox con un remoto vacío y dos clones:
  - (A) zip idéntico → borrado; zip distinto → `.local`; zip solo local → intacto; conflicto de BITACORA en el autostash → los dos lados, stash vacío;
  - (B) commit local que choca en BITACORA durante el rebase → resuelto y rebase continuado;
  - (C) conflicto en README → exit 2, rebase a medias sin tocar.
  Uso real aquí: borró la copia idéntica de `A-g3_30000k.zip`, trajo `14324fa` y reaplicó el autostash de BITACORA sin conflicto.

## 2026-11-06 21:03 · PC1 · A-g5

- Comando: `python student/train_agent.py --name A-g5 --init models/liga4/A-g4_4000k.zip --league-glob models/liga4/*.zip --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 30000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents {"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}`
- Pasos: 30,008,160 · inicio 2026-11-06T17:34 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 3.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 19:21 · PC3 · B-g2

- Comando: `python student/train_agent.py --name B-g2 --init models/league/B-g1_20000k.zip --reward agresiva --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_B_g2/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T17:59 · lr efectivo 0.0001
## 2026-11-06 21:10 · PC1 · pipeline: A-g5 → liga5 → A-g6

- Liga: `results/league_2026-11-06_2104_liga5.json` (15 modelos + baselines + bots).
- Elegido: `models/liga5/A-g5_24000k.zip` (mayor Elo principal de la rama A (2309)).
- Lanzado A-g6 (PID 899642): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g6 --init models/liga5/A-g5_24000k.zip --league-glob 'models/liga5/*.zip' --league-share 0.5 --pfsp hard --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g6.out 2>&1 &`
- Resumen: `results/pipeline_liga5.md`.

## 2026-11-06 22:32 · PC1 · A-g6

- Comando: `python student/train_agent.py --name A-g6 --init models/liga5/A-g5_24000k.zip --league-glob models/liga5/*.zip --league-share 0.5 --pfsp hard --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents {"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}`
- Pasos: 12,006,540 · inicio 2026-11-06T21:10 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 0 | 2 | 0 | 0% | 0.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 19:21 · PC3 · C-g2 (exploiter de A, superviviente)

- 6,003,270 pasos desde `C-g1_20000k.zip` contra `A-g4_4000k` (init 25 % vs A). Snapshots: C-g2_1500k: A 68 %, hard 100 %; C-g2_3000k: A 88 %, hard 90 %; C-g2_4500k: A 88 %, hard 100 %; C-g2_6000k: A 95 %, hard 90 %. Publicado: `C-g2_6000k.zip`. Tabla: `results/exploiters_C.md`.

## 2026-11-06 19:21 · PC3 · B-g2 (exploiter de A, agresiva)

- 6,003,270 pasos desde `B-g1_20000k.zip` contra `A-g4_4000k` (init 5 % vs A). Snapshots: B-g2_1500k: A 92 %, hard 90 %; B-g2_3000k: A 90 %, hard 100 %; B-g2_4500k: A 95 %, hard 100 %; B-g2_6000k: A 100 %, hard 100 %. Publicado: `B-g2_6000k.zip`. Tabla: `results/exploiters_B.md`.

## 2026-11-06 20:33 · PC3 · C-g3

- Comando: `python student/train_agent.py --name C-g3 --init models/league/C-g2_6000k.zip --reward superviviente --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_C_g3/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T19:22 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 0 | 2 | 0 | 0% | 0.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 20:34 · PC3 · C-g3 (exploiter de A, superviviente)

- 6,003,270 pasos desde `C-g2_6000k.zip` contra `A-g4_4000k` (init 95 % vs A). Snapshots: C-g3_1500k: A 75 %, hard 90 %; C-g3_3000k: A 95 %, hard 80 %; C-g3_4500k: A 88 %, hard 90 %; C-g3_6000k: A 88 %, hard 90 %. Publicado: `C-g3_6000k.zip`. Tabla: `results/exploiters_C.md`.

## 2026-11-06 20:34 · PC3 · B-g3

- Comando: `python student/train_agent.py --name B-g3 --init models/league/B-g2_6000k.zip --reward agresiva --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_B_g3/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T19:22 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 0 | 2 | 0 | 0% | 0.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 20:35 · PC3 · B-g3 (exploiter de A, agresiva)

- 6,003,270 pasos desde `B-g2_6000k.zip` contra `A-g4_4000k` (init 100 % vs A). Snapshots: B-g3_1500k: A 95 %, hard 100 %; B-g3_3000k: A 98 %, hard 100 %; B-g3_4500k: A 100 %, hard 100 %; B-g3_6000k: A 100 %, hard 100 %. Publicado: `B-g3_6000k.zip`. Tabla: `results/exploiters_B.md`.

## 2026-11-06 21:51 · PC3 · C-g4

- Comando: `python student/train_agent.py --name C-g4 --init models/league/C-g3_6000k.zip --reward superviviente --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_C_g4/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T20:34 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 1 | 1 | 0 | 50% | 0.50 |
| baseline:easy | 2 | 0 | 0 | 100% | 3.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: 

## 2026-11-06 21:51 · PC3 · B-g4

- Comando: `python student/train_agent.py --name B-g4 --init models/league/B-g3_6000k.zip --reward agresiva --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --eval-games 2 --league-glob models/pool_B_g4/*.zip --league-share 0.6 --pfsp hard --maps mixed2 --scenario-prob 0.3 --timesteps 6000000 --snapshot-every 1500000 --n-envs 3 --opponents {"baseline:hard":0.4,"my:stoch:baseline:hard":0.2,"baseline:medium":0.1,"my:bushcamper":0.1,"my:hitandhide":0.1,"aggressive":0.05,"defensive":0.05}`
- Pasos: 6,003,270 · inicio 2026-11-06T20:35 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 0 | 2 | 0 | 0% | 0.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| random | 1 | 1 | 0 | 50% | 1.00 |
| basic | 2 | 0 | 0 | 100% | 2.50 |
| aggressive | 2 | 0 | 0 | 100% | 1.50 |
| defensive | 1 | 1 | 0 | 50% | 0.50 |
| baseline:easy | 2 | 0 | 0 | 100% | 2.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 2.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 2.00 |

- Notas: 

## 2026-11-06 21:52 · PC3 · C-g4 (exploiter de A, superviviente)

- 6,003,270 pasos desde `C-g3_6000k.zip` contra `A-g4_4000k` (init 88 % vs A). Snapshots: C-g4_1500k: A 75 %, hard 90 %; C-g4_3000k: A 82 %, hard 80 %; C-g4_4500k: A 98 %, hard 90 %; C-g4_6000k: A 100 %, hard 90 %. Publicado: `C-g4_6000k.zip`. Tabla: `results/exploiters_C.md`.

## 2026-11-06 21:52 · PC3 · B-g4 (exploiter de A, agresiva)

- 6,003,270 pasos desde `B-g3_6000k.zip` contra `A-g4_4000k` (init 100 % vs A). Snapshots: B-g4_1500k: A 88 %, hard 100 %; B-g4_3000k: A 100 %, hard 90 %; B-g4_4500k: A 98 %, hard 100 %; B-g4_6000k: A 95 %, hard 100 %. Publicado: `B-g4_3000k.zip`. Tabla: `results/exploiters_B.md`.

## 2026-11-06 19:57 · PC2 · E-g1 (entrada automática de train_agent.py, que pone «PC1» fijo)

- Comando: `python student/train_agent.py --name E-g1 --init models/shared/A-g4_4000k.zip --league-glob models/shared/*.zip --league-share 0.5 --pfsp hard --reward nodraw --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 6 --opponents {"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}`
- Pasos: 12,006,540 · inicio 2026-11-06T18:28 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 3.00 |
| basic | 2 | 0 | 0 | 100% | 3.00 |
| aggressive | 2 | 0 | 0 | 100% | 3.00 |
| defensive | 2 | 0 | 0 | 100% | 3.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 1.00 |
| baseline:hard | 2 | 0 | 0 | 100% | 3.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 3.00 |

- Notas: ~2100 pasos/s con 6 entornos, 1,5 h. Durante el entrenamiento la RAM bajó a ~0,6 GB libres de 16: cada entorno ocupa ~600 MB porque carga los modelos de la liga. Con D-g1b a la vez, conviene no pasar de 6 entornos.

### Evaluación de los snapshots de E-g1 (`student/eval_snapshots.py`, nuevo)

- Determinista. 40 partidas por rival: los 16 mapas de official_aug y 4 aleatorios con semilla fija (20261106+k), por los dos lados. Modelo contra modelo es determinista, así que basta una partida por (mapa, lado).
  `python student/eval_snapshots.py --models "models/league/E-g1_*k.zip" --rivals model:models/shared/A-g4_4000k.zip model:models/shared/B-g1_20000k.zip model:models/shared/C-g1_20000k.zip baseline:hard --ref models/shared/A-g4_4000k.zip --out results/E-g1_eval --outbox E --procs 4`
  Salida en `results/E-g1_eval.md` y `.json` (con todas las partidas).

| modelo | A-g4_4000k | B-g1_20000k | C-g1_20000k | baseline:hard | total W-D-L |
|---|---|---|---|---|---|
| E-g1_2000k **← mejor** | 36-4-0 | 38-2-0 | 38-2-0 | 40-0-0 | 152-8-0 |
| E-g1_4000k | 32-8-0 | 36-4-0 | 32-8-0 | 38-2-0 | 138-22-0 |
| E-g1_6000k | 36-2-2 | 40-0-0 | 34-6-0 | 40-0-0 | 150-8-2 |
| E-g1_8000k | 38-2-0 | 40-0-0 | 32-8-0 | 40-0-0 | 150-10-0 |
| E-g1_10000k | 40-0-0 | 34-6-0 | 34-6-0 | 40-0-0 | 148-12-0 |
| E-g1_12000k | 36-0-4 | 40-0-0 | 34-6-0 | 36-4-0 | 146-10-4 |
| A-g4_4000k (ref) | 0-40-0 | 36-4-0 | 24-16-0 | 38-2-0 | 98-62-0 |

- Elegido **E-g1_2000k**: el que tiene menos empates (8) sin perder más que la referencia (0 derrotas). Empata en empates con 6000k, pero 6000k pierde 2. Copiado a `models/outbox/`, y sync lo publica.
- **Ojo:**
  - A-g4_4000k, B-g1_20000k y C-g1_20000k estaban en la liga de entrenamiento de E (`models/shared/*.zip`), y los mapas official_aug también están en mixed2. Es una evaluación contra rivales vistos, y probablemente sobreestima. Hay que confirmarlo en la liga con modelos que E no haya visto (B-g4, C-g4, A-g5, D-g1b…) y en mapas gen o aleatorios.
  - Con solo 2000k pasos (~15 min), la recompensa nodraw ya acaba con casi todos los empates contra A-g4: pasa de 40 empates a 4.
- Nota: el primer push de esta entrada (422d729) llevaba marcas de conflicto con las entradas de PC3. Se corrige en el commit siguiente, conservando las dos partes.

## 2026-11-06 23:50 · PC2 · D-g1b en pausa (apagado nocturno)

- Parado a 5,03M pasos (~6,5M desde la destilación, contando los 1,5M de D-g1). El último checkpoint es `models/checkpoints_D-g1b/D-g1b_5000000_steps.zip` (23:46, verificado: incluye `policy.optimizer.pth`). Snapshots para la liga: `D-g1b_2000k` y `D-g1b_4000k`.
  Ritmo medio en GPU con E-g1 en paralelo: ~0,9M pasos por hora.
- Curvas en `results/tb/D-g1b`. También se paró el sync, después de publicar `E-g1_2000k` a las 22:27.
- Para reanudar: `logs/run_D-g1c.ps1`, con la misma config, `--init` desde el checkpoint de 5M y `--device cuda`. Con `--init` el contador vuelve a 0, así que se usa el nombre D-g1c para no machacar los snapshots `D-g1b_*k`: D-g1c_Nk lleva en realidad 6,5M + N pasos.
## 2026-11-06 22:38 · PC1 · pipeline: A-g6 → liga6 → A-g7

- Liga: `results/league_2026-11-06_2233_liga6.json` (9 modelos + baselines + bots).
- Elegido: `models/liga6/A-g6_4000k.zip` (mayor Elo principal de la rama A (1828)).
- Lanzado A-g7 (PID 944689): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g7 --init models/liga6/A-g6_4000k.zip --league-glob 'models/liga6/*.zip' --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g7.out 2>&1 &`
- Resumen: `results/pipeline_liga6.md`.

## 2026-11-06 23:55 · PC1 · Pipeline en pausa hasta mañana

- Pipeline 3 (PID 798880) parado con `STOP_PIPELINE` mientras esperaba a A-g7 (generación 3/8). Hechas: liga5 → A-g6 y liga6 → A-g7. El archivo `STOP_PIPELINE` ya se ha borrado.
- A-g7 (PID 944689) se deja terminar solo (12M pasos, hacia las 00:05). El sync (PID 795544) sigue en marcha.
- Para reanudar donde se quedó (liga7 → A-g8 … A-g13; quedan 6 generaciones con la misma alternancia de PFSP, empezando por hard):
  `nohup .venv/bin/python student/pipeline.py --wait-pid logs/A-g7.pid --generations 6 --timesteps 12000000 --pfsp-modes hard,variance,hard,variance,hard,variance > logs/pipeline4.out 2>&1 &`

## 2026-11-07 00:00 · PC1 · pipeline.py: ramas desde archivo y reanudación tras un apagado

- `student/league_branches.txt` (B, C, D, X1, E; una por línea, con `#` para comentarios): `pipeline.py` lo lee **antes de cada liga**, así que para añadir una rama basta con editarlo, sin relanzar. Si no existe, usa esa misma lista por defecto. A y R1 se ignoran aunque aparezcan.
- Reanudación si se apaga el portátil antes de que A-g7 guarde `models/A-g7.zip`:
  - Sin modelo final ya se usaba el último snapshot (no hacía falta cambiar nada).
  - **Arreglado 1:** tras reiniciar, el PID de `logs/A-g7.pid` puede ser de otro proceso y el pipeline esperaría para siempre. Ahora comprueba en `/proc/<pid>/cmdline` que es `train_agent.py --name A-g7`; si no lo es, da el entrenamiento por terminado.
  - **Arreglado 2:** un snapshot o modelo final cortado a medias por el apagado (zip inválido) se descarta con aviso, en vez de romper la liga.
- Pruebas solo con archivos de mentira y `--dry-run`, sin lanzar nada:
  - sin archivo de ramas → lista por defecto; con «B, E» → solo B y E; se relee en cada llamada;
  - snapshot de 12M corrupto → se salta; sin `A-g7.zip` → último snapshot válido (10M); `A-g7.zip` corrupto → último snapshot;
  - PID vivo de otro proceso → no espera; proceso que sí es `train_agent.py --name A-g7` → espera a que acabe.
  - Con el PID real 944689, `is_training` = True.
- El comando para reanudar mañana no cambia.

## 2026-11-07 · PC2 · Evaluación de D y de los modelos nuevos de cada rama

- `student/eval_snapshots.py`, determinista, 40 partidas por pareja (16 official_aug + 4 aleatorios × 2 lados). Tabla completa en `results/eval_2026-11-07_D_y_nuevos.md` y `.json`. Total W-D-L sin contar el cruce consigo mismo:

| modelo | hard | contra los demás modelos (sin hard ni el cruce consigo mismo: 4 rivales; D, 5) |
|---|---|---|
| A-g6_4000k | 36-4-0 | 76-80-4 |
| C-g4_6000k | 38-2-0 | 72-74-14 |
| B-g4_3000k | 40-0-0 | 62-90-8 |
| A-g5_24000k | 40-0-0 | 30-52-78 |
| E-g1_2000k | 40-0-0 | 2-20-138 |
| D-g1b_5M (ckpt) | 8-30-2 | 6-22-172 |
| D-g1b_4000k | 12-22-6 | 0-8-192 |
| D-g1b_2000k | 6-34-0 | 0-22-178 |

- **D no cumple el criterio de CLAUDE.md** (≥ 50 % contra hard en determinista): su mejor resultado es 12-22-6 (30 %), y pierde casi todo contra los modelos nuevos. D-g1c (la continuación de D-g1b) quedó lanzado en GPU antes de esta evaluación.
- **E-g1_2000k no generaliza:** gana 40-0-0 a hard, pero pierde 0-0-40 contra A-g6, 2-10-28 contra A-g5, 0-8-32 contra B-g4 y 0-2-38 contra C-g4. El 152-8-0 de ayer era contra rivales de su propia liga (A-g4, B-g1, C-g1).
- **Arriba:** A-g6, C-g4 y B-g4 casi solo empatan entre ellos. A-g6 contra B y C: 4-34-2 y 6-32-2. B contra C: 8-30-2. El problema de los empates sigue entre los mejores. Son empates de verdad: al acabar el tiempo se decide por vida y después por daño, pero el daño de cada uno es la vida que ha perdido el otro, así que con la misma vida también hay el mismo daño.

## 2026-11-07 00:13 · PC1 · A-g7

- Comando: `python student/train_agent.py --name A-g7 --init models/liga6/A-g6_4000k.zip --league-glob models/liga6/*.zip --league-share 0.5 --pfsp variance --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents {"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}`
- Pasos: 12,006,540 · inicio 2026-11-06T22:38 · lr efectivo 0.0001
- Evaluación determinista (2 partidas/rival):

| rival | win | draw | lose | win rate | daño |
|---|---|---|---|---|---|
| random | 2 | 0 | 0 | 100% | 2.00 |
| basic | 2 | 0 | 0 | 100% | 2.50 |
| aggressive | 2 | 0 | 0 | 100% | 2.00 |
| defensive | 0 | 2 | 0 | 0% | 0.00 |
| baseline:easy | 2 | 0 | 0 | 100% | 1.00 |
| baseline:hard | 0 | 2 | 0 | 0% | 0.00 |
| baseline:medium | 2 | 0 | 0 | 100% | 2.00 |

- Notas: 

## 2026-11-07 12:45 · PC1 · Pipeline reanudado tras reinicio

- El portátil se reinició durante la noche; A-g7 ya había terminado (`models/A-g7.zip` a las 00:13). No quedaba nada en marcha: ni el pipeline ni el sync.
- **Corregido en `pipeline.py`:** el modelo final se guarda después del rollout que cruza el último snapshot, así que su `steps` es hasta ~8k mayor (A-g7: 12 006 540 frente a `A-g7_12000k`). Con la comparación exacta entraban los dos en la liga (también `A-g2_20000k` y `A-g2.zip` en ligas anteriores). Ahora se descarta el snapshot si está a menos de 10k pasos del final.
- Reanudado: pipeline **PID 6561** (`logs/pipeline.pid`, salida en `logs/pipeline4.out`), con liga7 → A-g8 … A-g13 y PFSP hard,variance,… :
  `nohup .venv/bin/python student/pipeline.py --wait-pid logs/A-g7.pid --generations 6 --timesteps 12000000 --pfsp-modes hard,variance,hard,variance,hard,variance > logs/pipeline4.out 2>&1 &`
  El PID guardado de A-g7 (944689) no correspondía a ningún proceso tras el reinicio y se detectó como terminado.
- Sync relanzado igual que antes: `nohup .venv/bin/python student/sync_models.py --loop --publish A > logs/sync.out 2>&1 &`, **PID 6564** (`logs/sync.pid`).

## 2026-11-07 12:48 · PC1 · pipeline: A-g7 → liga7 → A-g8

- Liga: `results/league_2026-11-07_1244_liga7.json` (9 modelos + baselines + bots).
- Elegido: `models/liga7/A-g7_8000k.zip` (mayor Elo principal de la rama A (1903)).
- Lanzado A-g8 (PID 7228): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /home/rubba/UNIVERSIDAD/ReinforcementLearning/Challenges/Challenge2_Tanks/.venv/bin/python student/train_agent.py --name A-g8 --init models/liga7/A-g7_8000k.zip --league-glob 'models/liga7/*.zip' --league-share 0.5 --pfsp hard --reward default --lr 1e-4 --clip-range 0.15 --ent-coef 0.02 --maps mixed2 --scenario-prob 0.3 --timesteps 12000000 --snapshot-every 2000000 --eval-games 2 --n-envs 10 --opponents '{"baseline:hard":0.2,"my:stoch:baseline:hard":0.15,"baseline:medium":0.15,"basic":0.05,"aggressive":0.05,"defensive":0.05,"my:rusher":0.05,"my:bushcamper":0.15,"my:hitandhide":0.15}' > logs/A-g8.out 2>&1 &`
- Resumen: `results/pipeline_liga7.md`.

## 2026-11-07 13:30 · PC1 · league.py: cuadro con la regla del README

- Regla del README: «best of 3 (draws do not count; tie-break by damage dealt)». La simulación anterior contaba los empates como partidas jugadas y, si la serie acababa igualada, tiraba una moneda: en la liga 7, el 44 % de las series de las rondas y el 71 % de las finales se decidían a cara o cruz.
- Ahora:
  - una partida empatada no cuenta y se juega la siguiente combinación de mapa y lado del mismo conjunto (oficiales en las rondas, aleatorios en la final), alternando lados, hasta que alguien gane 2 o se agoten las combinaciones distintas (8 y 16);
  - después gana quien tenga más victorias; si siguen iguales, el daño total de la serie; si también es igual, moneda.
- Columna nueva en la tabla de la liga, «series a moneda»: el % de las series de cada modelo que se deciden por moneda, más el total del cuadro en la nota.
- Pruebas con tablas sintéticas: las combinaciones siempre son distintas y con lados alternos (también con un número par de mapas); todo empate → moneda; una sola victoria → gana; solo empates pero más daño → gana por daño.
- Liga 7 recalculada con la regla nueva (mismas partidas, 20 000 sorteos):

| modelo | Elo | P(campeón) antes | P(campeón) ahora | series a moneda |
|---|---|---|---|---|
| A-g7_8000k | 1903 | 25,6 % | 12,7 % | 47 % |
| A-g6_4000k | 1898 | 13,0 % | 0,7 % | 43 % |
| A-g7_4000k | 1877 | 18,2 % | 5,4 % | 33 % |
| C-g4_6000k | 1863 | 2,9 % | 0,0 % | 0 % |
| B-g4_3000k | 1862 | 5,3 % | 2,7 % | 18 % |
| A-g7 (final) | 1814 | 35,0 % | **78,5 %** | 16 % |

  Series a moneda en todo el cuadro: 25,5 %. El A-g7 final no pierde ninguna partida contra el resto del cuadro y gana 2 de las 8 combinaciones oficiales contra casi todos: cuando los empates se repiten, eso le basta para ganar casi todas las series. A-g7_8000k empata las 8 combinaciones oficiales contra varios A y depende de la moneda en casi la mitad de sus series.
- El pipeline en marcha la usará a partir de la liga 8 (lanza `league.py` como proceso nuevo).
