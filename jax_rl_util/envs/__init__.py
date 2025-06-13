"""Environment Module."""

from .continuous_cartpole import CartPoleSwingUp, ContinuousCartPoleEnv  # noqa
from .dronegym import DroneGym  # noqa

BRAX_ENVS_POS_DIMS = {"ant": 2, "halfcheetah": 1, "humanoid": 2}
