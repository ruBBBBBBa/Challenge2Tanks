# CLAUDE.md · Challenge 2 Tanks

Contexto e instrucciones para Claude Code. Léelo entero antes de empezar.

Somos un equipo con 3 ordenadores (PC1, PC2, PC3) y en cada uno trabaja una sesión de Claude Code.
**Si no te han dicho qué ordenador eres, pregúntalo antes de hacer nada.**

---

## 1. Objetivo y plazo

- Entrenar con RL un `model.zip` que gane el torneo de clase contra los modelos de otros equipos (no solo contra los bots).
- Entrega: **martes 10/11/2026**. Se entrega `submissions/<equipo>/model.zip` más un informe.
- Torneo: eliminatoria con sorteo, al mejor de 3, en directo. Los lados se alternan. Los empates no cuentan y se desempata por daño causado.
  Las rondas se juegan en los 4 mapas oficiales y **la final en mapas secretos**.
- El torneo carga el modelo con `ModelAgent(..., deterministic=True)` y solo llama a `model.predict(obs)`.

## 2. Reglas que no se rompen

1. **No modificar nada en `tanks/` ni en `tools/`.** El torneo usa la versión original. Se puede importar de ahí y heredar clases en nuestro código.
2. Todo el código propio va en `student/` (o en scripts nuevos en la raíz).
3. El modelo entregado usa la observación oficial `Box(77,)` (`tanks.observations.vector_obs`) y `Discrete(6)`.
   Nada de frame stacking ni features propias. Se permite RecurrentPPO: el profesor confirma que `sb3-contrib` está instalado en el torneo.
4. Evaluar **siempre en determinista**, como el torneo.
5. **No usar MaskablePPO.** En el torneo `predict` va sin máscara y el modelo podría elegir acciones que nunca aprendió a evitar.
6. Las reglas y los mapas solo se cambian para entrenar (`TanksEnv(config=..., maps=...)`). El torneo usa los valores por defecto.
7. Recompensas, bots y callbacks a nivel de módulo (SubprocVecEnv los serializa). Mantener `if __name__ == "__main__":`.
8. Toca solo los archivos de los que es dueño tu ordenador (sección 7). Para cambiar otro, commit pequeño y avísalo en `BITACORA.md`.

## 3. Entorno

- `pip install -r requirements.txt "numpy<2" sb3-contrib`, manteniendo `gymnasium==0.29.1`.
- **`numpy<2` es obligatorio.** Los baselines se guardaron con Python 3.8, SB3 2.1.0, PyTorch 2.0.1 y numpy 1.24, y un zip guardado con numpy 2 puede no cargar en la máquina del profesor.
  `sb3-contrib` debe tener la misma versión menor que `stable-baselines3`.
- Ejecuta `python tools/checkenv.py` para comprobar el entorno y medir pasos/segundo. Apunta núcleos y pasos/s en `BITACORA.md`.
- Se entrena en CPU con `SubprocVecEnv`, con `n_envs` aproximadamente igual al número de núcleos físicos.
  Mantén `n_steps * n_envs` en torno a 8192, como en los baselines.

## 4. Lo que sabemos del código

**Juego** (`tanks/game.py`)
- Mapa de 20×20, movimientos simultáneos, 3 HP por tanque y sin respawn. Cooldown de disparo de 5 pasos.
- La bala sale y avanza 2 casillas en el mismo paso: a distancia 1 o 2 en línea **no se puede esquivar**. Dos balas que se cruzan se anulan.
- Un tanque en un arbusto es invisible si el enemigo está a más de 2 casillas (distancia de Chebyshev). Sus balas sí se ven.
- La partida acaba al destruir un tanque o a los 500 pasos. Entonces gana quien tenga más HP, después quien haya hecho más daño; si no, empate.
- Acciones: 0 noop, 1 forward, 2 backward, 3 turn_left, 4 turn_right, 5 fire. Girar gasta un paso; retroceder mantiene el cañón al frente.

**Observación** (`tanks/observations.py`)
- `[0:8]` yo; `[8:20]` enemigo (visible, offsets, orientación relativa, HP, cooldown, distancia, enemigo en mi mira, yo en su mira); `[20:52]` 8 rayos; `[52:76]` 3 balas enemigas; `[76]` tiempo (step/max_steps).
- Si el enemigo está oculto en un arbusto, sus campos de posición valen cero. Un MLP no tiene memoria.

**Entorno** (`tanks/tanks_env.py`)
- El rival vive dentro del entorno. Con `learner_side=None` se sortea el lado.
- `maps=None` equivale a los mapas oficiales más un 50 % de aleatorios. `maps` acepta nombres, rutas `.txt` o `"random"`.
- Al final del episodio, `info` incluye `result`, `is_success`, `hp`, `enemy_hp`, `damage_dealt`, `opponent` y `map`.
- `set_opponent(spec)` cambia de rival a partir del siguiente reset.

**Rivales** (`tanks/agents/base.py`)
- Specs disponibles: `random`, `basic`, `aggressive`, `defensive`, `param:...`, `baseline:easy|medium|hard`, `model:<ruta>.zip` (juega en determinista) y `my:<nombre>` (sale de `MY_BOTS` en `student/my_opponents.py`).
- Un dict `{spec: peso}` crea un `OpponentPool`: un rival por episodio. Su `name` es `pool[<spec>]`, así que se puede saber contra quién se jugó.
- `ModelAgent(path, deterministic=True|False)` maneja PPO, A2C, DQN y sb3-contrib, incluido el estado de la LSTM. Su `__getstate__` no serializa el modelo, que se carga de forma perezosa en cada subproceso.
- `_load_student_bot` hace `name not in MY_BOTS` y luego `MY_BOTS[name]()`. Un `dict` propio que sobrescriba `__contains__` y `__getitem__` permite aceptar specs como `my:stoch:<ruta>` → `ModelAgent(ruta, deterministic=False)`.

**Utilidades**
- `tanks.match.play_match(a0, a1, game_map, seed)` devuelve `winner`, `damage`, `steps`, `end_reason`, etc. La partida es determinista si los dos agentes lo son.
- También están `tanks.evaluation.evaluate_agent`, `tanks.WinRateCurriculum`, `tanks.maps.generator.random_map(rng, n_features=...)` y `GameMap.from_ascii`.

**Baselines** (`tanks/agents/baselines/*.zip`)
- PPO con `MlpPolicy` `[256, 256]`, γ = 0.995, gae_lambda 0.95, n_steps 1024, 8 entornos, batch 2048, n_epochs 10 y ent_coef 0.01.
- Pasos de entrenamiento: easy 0.3M, medium 1.5M y hard 5M.

## 5. Estrategia acordada

1. **Partimos de `baseline:hard`**, con permiso del profesor. Es la base común y todas las ramas se hacen con fine-tuning desde `tanks/agents/baselines/hard.zip`.
2. **Ramas:**

   | ID | Punto de partida | Recompensa | Ordenador |
   |---|---|---|---|
   | A | hard | generalista | PC1 |
   | B | hard | agresiva | PC3 |
   | C | hard | superviviente | PC3 |
   | D | hard destilado en una LSTM (RecurrentPPO) | generalista | PC2 |
   | R1 | hard, otra semilla | variante propia | por decidir. **Nunca entra en ningún pool**: rival reservado solo para evaluar |

3. **Liga.** Cada mañana se juega un todos contra todos determinista (`student/league.py`). En cada generación los agentes entrenan contra esta mezcla:
   - ~50 % copias congeladas de la liga, con pesos PFSP;
   - ~20 % su propia versión más reciente;
   - ~30 % bots y baselines, para no olvidar lo aprendido.
4. **Explotadores.** Son agentes entrenados solo contra el líder, o contra hard, para encontrar sus fallos. Entran al pool, pero no se entregan.
5. **Selección final** (lunes 9 a las 12:00). Se decide con el peor resultado contra el top-3 de la liga, el bracket simulado, el rendimiento en mapas aleatorios y el resultado contra R1. No vale la media.

**Riesgo conocido:** todo desciende de hard, igual que probablemente muchos modelos de otros equipos. La diversidad sale de las recompensas, de D, de R1 y de los explotadores.

## 6. Herramientas a construir

### `student/envs.py` (PC1)
`StudentTanksEnv(TanksEnv)` con:
- `set_pool_weights(weights)`. Si el rival es un `OpponentPool`, llama a `set_weights` sin recrear el pool, para no perder la caché de modelos cargados. Si no, llama a `set_opponent`.
- `set_reward_param(name, value)`. Lo reenvía a `self.reward_fn`, para mezclar o apagar el shaping desde un callback.
- Conjuntos de mapas:
  - `official`;
  - `official_aug`: cada mapa oficial con sus variantes original, espejo horizontal, rotación de 90° y rotación de 90° + espejo. Las demás variantes coinciden por la simetría. Comprueba que siguen siendo simétricos respecto al centro y que los spawns son válidos, y guárdalos como `.txt` en `student/maps/`;
  - `random`;
  - `mixed`: 50 % de `official_aug` y 50 % de aleatorios.

### `student/my_opponents.py` (PC1)
- Specs estocásticos `my:stoch:<ruta.zip>` (también `my:stoch:baseline:hard`), con un dict propio en `MY_BOTS`. Usa siempre barras `/` en las rutas.
- Bots propios, reutilizando los helpers de `tanks/agents/heuristics.py`:
  - `bushcamper`: va al arbusto más cercano, espera y dispara cuando el enemigo entra en su línea. Si va ganando en HP, no sale.
  - `rusher`: va directo al enemigo por BFS y dispara en cuanto está alineado.
  - `hitandhide`: busca un impacto y después se esconde, lejos o en un arbusto, hasta el final.

### `student/my_rewards.py` (PC3)
- Una clase de recompensa con dos parámetros ajustables:
  - `blend` (de 0 a 1): mezcla entre `default_reward` y la recompensa nueva. Así no se rompe de golpe la red de valor de hard.
  - `shaping_scale`: para ir apagando el shaping.
- `REWARDS = {"default": ..., "generalista": ..., "agresiva": ..., "superviviente": ...}`, todo a nivel de módulo.
- **Shaping basado en potencial:** F = γ·Φ(s′) − Φ(s), con γ = 0.995. Φ se calcula solo con datos del estado: distancia cuando el enemigo es visible, enemigo en mi mira con el arma cargada, estar en la línea de una bala con amenaza, etc.
- **Agresiva:** más peso a `hits_dealt`, bonus por tener al enemigo en la mira mientras él recarga y empate más castigado.
- **Superviviente:** más castigo a `hits_taken`, bonus por salir de la línea de una bala con amenaza y castigo por estar en la mira del enemigo con mi cooldown > 1.
- Prueba cada recompensa 50k pasos desde hard y mira replays buscando trampas: apuntar sin disparar, esconderse hasta el final, girar en bucle.

### `student/train_agent.py` (PC1)
Un único CLI para todos los entrenamientos:

```
--name  --algo {ppo,recurrent}  --init <ruta|baseline:hard|none>  --reward <nombre>
--opponents <json mezcla> | --stages <json curriculum>
--league-glob "models/league/*.zip"  --league-share 0.5  --pfsp {off,variance,hard}
--maps {official,official_aug,random,mixed}  --timesteps  --n-envs  --lr  --ent-coef
--clip-range  --seed  --blend-steps 300000  --snapshot-every 250000
```

- Tiene que funcionar con `--reward default` aunque `my_rewards.py` todavía no tenga las recompensas nuevas.
- Fine-tuning desde `hard.zip`: `PPO.load(path, env=env, custom_objects={"learning_rate": lr, "clip_range": clip})`. **Comprueba que el LR efectivo es el nuevo** (por defecto 1e-4). El resto de hiperparámetros, como hard.
- Callbacks:
  - mezcla de recompensa (`blend` de 0 a 1 durante `--blend-steps`);
  - PFSP;
  - snapshots en `models/league/<name>_<k>k.zip`;
  - checkpoints en `models/checkpoints_<name>/`;
  - `WinRateCurriculum`, si se pasan `--stages`.
- Guarda `models/<name>.zip` y `models/<name>.json` con todos los argumentos, la semilla, los pasos, la fecha y el ordenador.
- TensorBoard en `logs/` con `tb_log_name=<name>`. Monitor con `info_keywords=("is_success",)`.
- Al terminar, hace una evaluación determinista contra bots y baselines y añade una entrada en `BITACORA.md`.

### `student/pfsp.py` (PC1)
- Es un callback que lleva el win rate contra cada rival a partir de `info["opponent"]` (`pool[<spec>]`) al final de cada episodio, con prior Beta(1,1).
- Cada ~200 episodios recalcula los pesos de la parte de liga:
  - `variance` → p(1−p): da más peso a los rivales parejos. Para el principio.
  - `hard` → (1−p)²: da más peso a los rivales que nos ganan. Para el final.
- Peso mínimo 0.02. La parte fija (bots y baselines) conserva su cuota.
- Aplica los pesos con `env_method("set_pool_weights", w)` y registra en TensorBoard el win rate contra cada rival.

### `student/league.py` (PC1)
- Participantes: glob de modelos, bots y baselines (`--models`, `--bots`, `--baselines`).
- Todas las partidas son deterministas.
  - **Modelo contra modelo es determinista, así que basta una partida por (mapa, lado).**
  - Contra bots, que tienen azar, se juegan varias semillas.
- Mapas: los 4 oficiales, `official_aug` y N aleatorios con semillas fijas (8 por defecto). Los aleatorios se reportan aparte, porque son la referencia para la final.
- En paralelo con `multiprocessing`.
- Salida en `results/league_<fecha>.md` y `.json` con:
  - ranking por Elo (Bradley–Terry);
  - win rate contra el top-3 y peor emparejamiento;
  - win rate en mapas aleatorios;
  - empates y daño medio;
  - partidas sospechosas: timeout con 0 daño, que indica un bucle.
- Simulación de bracket: eliminatoria de 8 al mejor de 3, alternando lados y con el empate decidido por daño. Muchas simulaciones con sorteos aleatorios para estimar la probabilidad de campeón de cada modelo.
- `--replays-dir` para guardar replays de las derrotas de los mejores modelos.

### `student/distill.py` (PC2): agente D
1. **Generar datos con hard**, en determinista y con muestreo, contra una mezcla de bots, hard y medium, en mapas `mixed` y en ambos lados.
   En cada paso se guardan la observación, las probabilidades de acción de hard, la recompensa (por defecto) y si termina el episodio. Objetivo: entre 1M y 2M pasos.
2. **Destilar.** Crear un `RecurrentPPO("MlpLstmPolicy", ...)` y entrenarlo por secuencias (BPTT truncado, ~128 pasos):
   - la parte actor, con KL hacia las probabilidades de hard;
   - la parte crítica, con retornos (γ = 0.995), para que la red de valor no empiece aleatoria.
3. Guardar `models/D-distill.zip`. Comprobar que carga con `tools/evaluate.py` y que juega parecido a hard (win rate similar contra los bots).
4. Seguir con RL: `train_agent.py --algo recurrent --init models/D-distill.zip`, con LR bajo.

**Límite:** si el viernes 6 a las 18:00 D no gana al menos el 50 % contra hard en determinista, se abandona y PC2 pasa a entrenar R1 y el explotador anti-hard.

## 7. Encargo de cada ordenador

| Ordenador | Dueño de | Esta noche (jueves 5) |
|---|---|---|
| PC1 | `envs.py`, `train_agent.py`, `pfsp.py`, `my_opponents.py`, `league.py` | 1) `envs.py` + `train_agent.py` + specs estocásticos, con push lo antes posible porque PC3 depende de ello. 2) Lanzar A desde hard. 3) `pfsp.py` y bots propios para la liga del viernes. |
| PC2 | `distill.py` | Medir velocidad. Escribir `distill.py`, generar datos y destilar. Lanzar el RL de D por la noche. |
| PC3 | `my_rewards.py` | 1) `my_rewards.py` y push (PC1 lo importa). 2) Pruebas de 50k pasos. 3) Lanzar B y C desde hard, repartiendo los núcleos. (`league.py` pasa a PC1 el viernes 6.) |

**Mezcla inicial de rivales** (generación 1):

```json
{"baseline:hard": 0.25, "my:stoch:baseline:hard": 0.20, "baseline:medium": 0.10,
 "basic": 0.10, "aggressive": 0.10, "defensive": 0.10,
 "my:bushcamper": 0.075, "my:rusher": 0.075}
```

Si los bots propios aún no existen, reparte su peso entre `basic`, `aggressive` y `defensive`.

**Hiperparámetros de partida para el fine-tuning:**
- lr 1e-4 y clip 0.1–0.2;
- ent_coef 0.01 en A, y 0.015–0.02 en B y C para separarlas más de hard;
- el resto, como hard;
- mapas: `mixed`.

## 8. Calendario

| Día | Qué |
|---|---|
| Jue 5 (noche) | A, B y C desde hard. D destilado y en RL. |
| Vie 6 | Por la mañana, liga 1: elegir principales y lanzar la generación 2 con PFSP. A las 18:00, decisión sobre D. |
| Sáb 7 | Liga 2. Generación 3. Explotador contra el líder. |
| Dom 8 | Liga 3 con más mapas aleatorios. Generación 4 o ajuste en mapas aleatorios. Empezar el informe. |
| Lun 9 | Liga final. Congelar el candidato a las 12:00. Evaluación contra R1 y en mapas aleatorios, carga en entorno limpio y GIFs. |
| Mar 10 | Entrega. |

## 9. Nombres, git y registro

- Modelos: `models/<ID>.zip` + `models/<ID>.json`. El ID es rama + generación: `A-g1`, `B-g2`, `D-distill`, `X1-antihard`, `R1`.
- Snapshots para la liga en `models/league/`. Sube a git **solo los que se usan** (2 o 3 por entrenamiento), no todos los checkpoints.
  `logs/` y `models/checkpoints_*` están en `.gitignore`.
- `git pull --rebase` antes de empezar cada tarea. Commits pequeños con mensajes claros.
- **`BITACORA.md`:** una entrada por experimento con fecha y hora, ordenador, ID, comando exacto, pasos, resultados (evaluate o league) y qué falló o qué se aprendió.
  El equipo lo pasa al documento compartido.

## 10. Antes de dar un modelo por bueno, y antes de entregar

- `python tools/evaluate.py --agent models/<ID>.zip --games 40`
- Ver 3–5 partidas en determinista (`tools/play_match.py ... --no-render --gif out/x.gif` o `--save-replay replays/`), buscando bucles, empates sin daño y trampas.
- Entrega:
  1. Copiar el modelo a `submissions/<equipo>/model.zip`.
  2. Ejecutar `python tools/evaluate.py --agent submissions/<equipo>/model.zip` en un entorno virtual limpio con `requirements.txt`, `numpy<2` y `sb3-contrib`.
- Para el informe: guardar curvas de TensorBoard, tablas de la liga y GIFs, y dejar documentado que partimos de `baseline:hard`.

## REGLAS IMPORTANTES
Entrenamientos largos: lánzalos siempre desacoplados de la sesión (nohup o tmux en Linux/WSL; Start-Process en Windows), con la salida en logs/<ID>.out. Al lanzarlos, dime el PID y el comando para seguir el progreso.

Rendimiento: lanza siempre con OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, o pon torch.set_num_threads(1) en cada subproceso. Sin eso va unas 26 veces más lento (36 frente a 950 fps en PC1). También afecta a league.py y distill.py, que lanzan procesos que cargan modelos. Snapshots para la liga: como mínimo cada 2M pasos.
**Curvas de TensorBoard:** al terminar o parar cada entrenamiento, copia su carpeta `logs/<ID>_1/` a `results/tb/<ID>/` (si se relanzó con el mismo nombre, la de `_2`) y súbela con `git add results/tb/<ID>`. `logs/` está en `.gitignore`: si no se copia, las curvas solo existen en el ordenador que entrenó.
