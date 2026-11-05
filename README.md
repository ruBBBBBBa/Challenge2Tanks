# Challenge 2 — Tanks

Two tanks, one arena. Train a reinforcement learning agent that beats your classmates.

You train against opponents that live **inside** the environment (heuristic bots, teacher
baselines, your own frozen models, your own bots), so for Stable-Baselines3
it is a normal single-agent Gymnasium environment, just like the Snake challenge.
The real versus happens in the final tournament: your `model.zip` against everybody else's.

```bash
pip install -r requirements.txt
python tools/checkenv.py                              # sanity checks
python tools/play_human.py --opponent basic           # play yourself (W/A/S/D + Space)
python train_simple.py                                # simplest PPO vs "basic" (like the Snake one)
python train.py                                       # PPO vs "basic", parallel envs, your reward
python tools/evaluate.py --agent models/tanks_ppo.zip
python tools/play_match.py --p0 models/tanks_ppo.zip --p1 aggressive
```

All commands are run from the `Challenge2_Tanks/` folder. Works on Windows, Linux and macOS
(the scripts that open a window need a display: use `--no-render` / `--gif` on a server).

## 0. Folder structure

```
Challenge2_Tanks/
├── train_simple.py      ✏️ simplest training (create env + default PPO)
├── train.py             ✏️ PPO vs basic with parallel envs, uses student/my_rewards.py
├── student/             ✏️ YOUR code, change anything here
│   ├── my_rewards.py        your reward function (used by train.py and train_advanced.py)
│   ├── my_opponents.py      your own bots -> opponent spec "my:<name>"
│   └── train_advanced.py    curriculum, mixtures, self-play, algorithms, hyperparameters
├── tanks/               🔒 LOCKED engine: game rules, env, official observation, bots, maps
├── tools/               🔒 LOCKED scripts: checkenv, evaluate, play_match, play_human, replay
├── models/              (created) your trained models
└── submissions/         final uploads
```

🔒 **Do not modify `tanks/` or `tools/`.** The tournament uses the original version, so any change
there (rules, observation, bots) would only make your training different from the real games.
Read them as much as you want, and import from them in your own code.

---

## 1. The game

- 20×20 grid, **simultaneous moves**, 2 tanks with **3 HP** each, no respawn.
- **Bullets fly 2 cells per step**: they can be dodged, and two bullets that meet cancel each other.
- **Fire cooldown: 5 steps.**
- Tiles: `#` wall (indestructible), `B` brick (2 hits to destroy), `~` water (blocks tanks, not bullets),
  `*` bush (a tank inside is invisible unless the enemy is within 2 cells).
- A game ends when a tank is destroyed, or after **500 steps**: then more HP wins, then more damage dealt,
  otherwise it is a draw.
- Maps are point-symmetric. Official maps are in `maps/`; `"random"` generates a new symmetric map.
  The final match of the tournament is played on **secret maps**, so do not overfit to the official ones.

### Actions — `Discrete(6)` (relative to the tank's facing)

| 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| noop | forward | backward | turn left | turn right | fire |

Turning takes a step and does not move the tank. Backward keeps the gun pointing forward.

### Observation — `Box(77,)` (official, fixed)

Built by `tanks.observations.vector_obs(state)`, mostly in the tank's own frame:

| Slice | Content |
|---|---|
| `[0:8]` | my position, facing (one-hot), HP, cooldown |
| `[8:20]` | enemy: visible, forward/right offset, relative facing, HP, cooldown, distance, *enemy in my sights*, *I am in its sights* |
| `[20:52]` | 8 rays (forward, forward-right, …): distance to wall/brick, is brick, distance to non-drivable cell, enemy on the ray |
| `[52:76]` | 3 nearest enemy bullets: present, offset, relative direction, *threat* |
| `[76]` | time (step / 500) |

> **The tournament feeds your model exactly this observation.** You cannot use a different
> observation (no frame stacking, no custom features) in your submitted model.

### Rewards — `student/my_rewards.py`  ← your main design lever

```python
def my_reward(prev_state, state, events) -> float: ...     # in student/my_rewards.py
env = TanksEnv(opponent="basic", reward_fn=my_reward)
```

`events` contains `shots_fired, hits_dealt, hits_taken, bricks_destroyed, bumped, bullets_cancelled,
result ("win"/"lose"/"draw"/None), end_reason`. `state` is the full egocentric game state
(map, both tanks, bullets) — see `TanksGame.get_state` in `tanks/game.py`. The default is
`tanks.rewards.default_reward`; `WeightedReward(...)` lets you change weights; `approach_and_aim_reward`
is an example of shaping.

### Episode end (Gymnasium API)

`obs, reward, terminated, truncated, info = env.step(action)`
- `terminated=True` when a tank is destroyed, `truncated=True` at the 500-step limit.
- At the end, `info` has `result`, `end_reason`, `is_success` (1.0 for a win), `hp`, `enemy_hp`, `damage_dealt`.

---

## 2. Opponents

Every opponent is an *agent spec*, accepted by `TanksEnv(opponent=...)`, `play_match.py` and `evaluate.py`:

| Spec | Opponent |
|---|---|
| `"random"` | random actions |
| `"basic"`, `"aggressive"`, `"defensive"` | heuristic bots (open source: `tanks/agents/heuristics.py`) |
| `"param:aggression=0.8,caution=0.2,trigger=1.0,noise=0.05,flank=0.5"` | your own mix of the heuristic parameters |
| `"baseline:easy"`, `"baseline:medium"`, `"baseline:hard"` | teacher PPO models (`tanks/agents/baselines/`) |
| `"my:camper"` | your own bot registered in `MY_BOTS` (`student/my_opponents.py`) |
| `"model:models/my_v1.zip"` | **your own frozen model** → manual self-play: train v1, train v2 against v1, … |
| `{"basic": 0.4, "aggressive": 0.3, "model:models/v1.zip": 0.3}` | **mixture**: one opponent sampled per episode |

You can also write your own bots in `student/my_opponents.py`: subclass `tanks.agents.Agent`,
implement `act(state) -> int` (`tanks.game.valid_actions(state)` and the helpers in
`tanks/agents/heuristics.py` are useful) and register it in `MY_BOTS` to use it as `"my:<name>"`.

**Curriculum** — `tanks.WinRateCurriculum` (already wired in `student/train_advanced.py`) moves to the next opponent (or mixture)
when your win rate goes over a threshold:

```python
from tanks import WinRateCurriculum
stages = ["random", "basic", {"basic": 0.5, "aggressive": 0.5}, {"defensive": 0.5, "baseline:easy": 0.5}]
model.learn(1_000_000, callback=WinRateCurriculum(stages, threshold=0.7, window=100))
```

---

## 3. What you can change

- ✅ Reward function (`reward_fn`)
- ✅ Algorithm and hyperparameters (PPO, A2C, DQN; `sb3-contrib` algorithms such as QRDQN, TRPO, RecurrentPPO also load)
- ✅ Network architecture (`policy_kwargs`)
- ✅ Opponents, mixtures, curriculum, self-play with frozen copies, your own bots
- ✅ Everything in `student/`, `train.py`, `train_simple.py`, and your own new files
- ✅ Game rules **for training** (`TanksEnv(config={"max_steps": 300, ...})`, `maps=[...]`) — the tournament uses the defaults
- ✅ Action masking / safety filters **during training** (`env.valid_actions()`, `game.valid_actions(state)`)
- ❌ The observation of the submitted model (always `vector_obs`, `Box(77,)`)
- ❌ Hand-written logic in the submission: only `model.predict(obs)` is used
- ❌ Modifying `tanks/` or `tools/`

---

## 4. Submission

`submissions/<your_name>/model.zip` + a report (delivered separately).

Check your model before submitting: `python tools/evaluate.py --agent submissions/<your_name>/model.zip`.
A model that does not load or does not use the official observation/action spaces is disqualified.

## 5. Tournament and grading

- Single-elimination bracket, **best of 3** (draws do not count; tie-break by damage dealt),
  random draw, played live in class.
- Rounds on the official maps, **the final on secret maps**.
- Grade: **tournament position** + **report** (weights announced by the teacher).

The report should explain: reward design, opponents/curriculum used and why, algorithm and
hyperparameters, training curves (TensorBoard), evaluation against the bots and baselines,
what failed, and how to reproduce your model.

---

## 6. Files

| File | |
|---|---|
| `train_simple.py` | Simplest training: create the env and a default PPO |
| `train.py` | PPO vs basic with parallel envs (`--timesteps --n-envs --opponent --name`) |
| `student/my_rewards.py` | Your reward |
| `student/my_opponents.py` | Your bots (`"my:<name>"`) |
| `student/train_advanced.py` | Curriculum, mixtures, self-play, algorithm, hyperparameters |
| `tanks/game.py` | Pure game rules (`TanksGame`, `get_state`, `valid_actions`) |
| `tanks/tanks_env.py` | Gymnasium env with the opponent inside |
| `tanks/observations.py` | Official observation |
| `tanks/rewards.py` | Default reward + `WeightedReward` + shaping example |
| `tanks/curriculum.py` | Win-rate curriculum callback |
| `tanks/maps/` | Official maps (ASCII) + random symmetric generator |
| `tanks/agents/` | Agent interface, random, heuristics, SB3 model wrapper, opponent pool, baselines |
| `tools/evaluate.py` | Win rate vs bots and baselines |
| `tools/play_match.py` | Watch two agents, save GIFs/replays |
| `tools/play_human.py` | Play with the keyboard |
| `tools/replay.py` | Watch a replay / export a GIF |
| `tools/checkenv.py` | Env checks + speed |

Tips: train with `SubprocVecEnv` (see `train.py`, keep the `if __name__ == "__main__":`
guard on Windows), always evaluate against several opponents, and watch your agent play —
replays reveal exploits that numbers hide.
