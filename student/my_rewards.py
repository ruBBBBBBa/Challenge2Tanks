"""
Recompensas del equipo (PC3). Ver CLAUDE.md, sección 6.

    REWARDS = {"default", "generalista", "agresiva", "superviviente"}

Cada recompensa nueva es una clase con dos parámetros ajustables desde un callback
(`env_method("set_reward_param", name, value)`):

  blend          0..1  r = (1 - blend) * default_reward + blend * r_nueva.
                       train_agent.py lo sube de 0 a 1 durante --blend-steps para no
                       romper de golpe la red de valor de hard.
  shaping_scale  >= 0  multiplica el shaping; ponerlo a 0 lo apaga.

r_nueva = eventos con pesos propios + bonus de eventos + shaping_scale * F
F       = GAMMA * Phi(s') - Phi(s)   (shaping basado en potencial, GAMMA = 0.995 como hard)

Phi solo usa datos del estado (egocéntrico) y vale 0 en el estado terminal, así que no
cambia la política óptima: quedarse quieto "apuntando sin disparar" o girar en bucle no
acumula recompensa (en un estado con Phi > 0 quedarse da (GAMMA - 1) * Phi < 0).

Términos de Phi (todos opcionales según los pesos de cada rama):
  dist      -d_dist * distancia Manhattan normalizada al enemigo (si no se ve, distancia máxima)
  aim       +d_aim si el enemigo está en mi mira con línea despejada y mi arma cargada
  aim_rel   +d_aim_rel si además el enemigo está recargando (cooldown > 0)        [agresiva]
  threat    -d_threat si estoy en la línea de una bala enemiga que viene hacia mí
            (salir de esa línea da +d_threat)                                       [superviviente]
  exposed   -d_exposed si estoy en la mira del enemigo con mi cooldown > 1          [superviviente]

Todo a nivel de módulo: SubprocVecEnv serializa la recompensa.
"""
from tanks.game import clear_line, direction_to
from tanks.rewards import DEFAULT_WEIGHTS, WeightedReward, default_reward

GAMMA = 0.995
THREAT_RANGE = 6          # casillas: una bala avanza 2 por paso, así que son ~3 pasos de aviso


# ----------------------------------------------------------------------------
# Rasgos del estado
# ----------------------------------------------------------------------------
def enemy_distance(state):
    """Distancia Manhattan al enemigo, o None si está oculto."""
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return None
    return abs(me["x"] - en["x"]) + abs(me["y"] - en["y"])


def enemy_in_my_sights(state):
    """Miro hacia el enemigo visible con línea de tiro despejada."""
    me, en = state["me"], state["enemy"]
    return bool(en["visible"]
                and direction_to(me["x"], me["y"], en["x"], en["y"]) == me["dir"]
                and clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"]))


def me_in_enemy_sights(state):
    """El enemigo visible mira hacia mí con línea de tiro despejada."""
    me, en = state["me"], state["enemy"]
    return bool(en["visible"]
                and direction_to(en["x"], en["y"], me["x"], me["y"]) == en["dir"]
                and clear_line(state["tiles"], en["x"], en["y"], me["x"], me["y"]))


def bullet_threat(state):
    """Hay una bala enemiga a <= THREAT_RANGE casillas que viene hacia mí sin obstáculos."""
    me = state["me"]
    for b in state["bullets"]:
        if b["mine"]:
            continue
        if direction_to(b["x"], b["y"], me["x"], me["y"]) != b["dir"]:
            continue
        if abs(b["x"] - me["x"]) + abs(b["y"] - me["y"]) > THREAT_RANGE:
            continue
        if clear_line(state["tiles"], b["x"], b["y"], me["x"], me["y"]):
            return True
    return False


# ----------------------------------------------------------------------------
# Recompensa con blend y shaping
# ----------------------------------------------------------------------------
NO_SHAPING = dict(dist=0.0, aim=0.0, aim_rel=0.0, threat=0.0, exposed=0.0)


class ShapedReward:
    """Eventos con pesos propios + shaping basado en potencial, mezclados con default_reward."""

    weights = DEFAULT_WEIGHTS
    shaping = NO_SHAPING

    def __init__(self, blend=1.0, shaping_scale=1.0):
        self.blend = float(blend)
        self.shaping_scale = float(shaping_scale)
        self.events_reward = WeightedReward(self.weights)
        self.phi_weights = dict(NO_SHAPING, **self.shaping)

    # Phi(s): solo depende del estado
    def potential(self, state):
        w = self.phi_weights
        me, en = state["me"], state["enemy"]
        phi = 0.0
        if w["dist"]:
            d = enemy_distance(state)
            dmax = state["width"] + state["height"] - 2
            phi -= w["dist"] * (dmax if d is None else d) / dmax
        if (w["aim"] or w["aim_rel"]) and me["cooldown"] == 0 and enemy_in_my_sights(state):
            phi += w["aim"]
            if en["cooldown"]:
                phi += w["aim_rel"]
        if w["exposed"] and me["cooldown"] > 1 and me_in_enemy_sights(state):
            phi -= w["exposed"]
        if w["threat"] and bullet_threat(state):
            phi -= w["threat"]
        return phi

    def shaping_term(self, prev_state, state, events):
        phi1 = 0.0 if events["result"] is not None else self.potential(state)
        return GAMMA * phi1 - self.potential(prev_state)

    def new_reward(self, prev_state, state, events):
        r = self.events_reward(prev_state, state, events)
        if self.shaping_scale:
            r += self.shaping_scale * self.shaping_term(prev_state, state, events)
        return r

    def __call__(self, prev_state, state, events):
        b = self.blend
        r = 0.0
        if b < 1.0:
            r += (1.0 - b) * default_reward(prev_state, state, events)
        if b > 0.0:
            r += b * self.new_reward(prev_state, state, events)
        return float(r)


class GeneralistaReward(ShapedReward):
    """Rama A: los pesos por defecto con un shaping suave de acercarse, apuntar y no comerse balas."""

    weights = dict(DEFAULT_WEIGHTS)
    shaping = dict(dist=0.3, aim=0.10, threat=0.10, exposed=0.05)


class AgresivaReward(ShapedReward):
    """Rama B: premia mucho el daño, castiga más el empate y apuntar cuando el enemigo recarga."""

    weights = dict(DEFAULT_WEIGHTS, **{
        "hits_dealt": 3.0,
        "hits_taken": -1.5,
        "draw": -5.0,
        "shots_fired": -0.01,
        "step": -0.01,
    })
    shaping = dict(dist=0.5, aim=0.10, aim_rel=0.15, threat=0.05)


class SupervivienteReward(ShapedReward):
    """Rama C: castiga más recibir daño, premia salir de la línea de una bala y no estar expuesto recargando."""

    weights = dict(DEFAULT_WEIGHTS, **{
        "hits_dealt": 2.0,
        "hits_taken": -3.5,
        "draw": -2.0,
        "step": -0.003,
    })
    shaping = dict(dist=0.1, aim=0.10, threat=0.30, exposed=0.20)


# train_agent.py instancia las clases (una instancia por entrenamiento)
REWARDS = {
    "default": default_reward,
    "generalista": GeneralistaReward,
    "agresiva": AgresivaReward,
    "superviviente": SupervivienteReward,
}

# compatibilidad con train.py / train_advanced.py, que importan my_reward
my_reward = GeneralistaReward()

__all__ = ["REWARDS", "my_reward", "ShapedReward", "GeneralistaReward", "AgresivaReward",
           "SupervivienteReward", "GAMMA"]
