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
