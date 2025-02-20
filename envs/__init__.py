"""Environment Module."""
import gymnasium

from .continuous_cartpole import CartPoleSwingUp, ContinuousCartPoleEnv  # noqa
from .dronegym import DroneGym  # noqa

gymnasium.register(id="CartpoleContinuousJax-v0", entry_point=ContinuousCartPoleEnv, order_enforce=False)
gymnasium.register(id="CartpoleContinuousJaxSwingUp-v0", entry_point=CartPoleSwingUp, order_enforce=False)
BRAX_ENVS_POS_DIMS = {"ant": 2, "halfcheetah": 1, "humanoid": 2}