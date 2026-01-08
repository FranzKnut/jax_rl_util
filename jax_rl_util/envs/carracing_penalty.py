"""A version of CarRacing environment with penalty for going off track."""

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
import numpy as np


class CarRacingPenaltyEnv(CarRacing):
    """A version of CarRacing environment with penalty for going off track."""

    def __init__(self, penalty_coeff: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.penalty_coeff = penalty_coeff

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Apply penalty for going off track
        positions = np.array([w.position for w in self.car.wheels])
        distances = np.array(self.track)[:, -2:][:, None] - positions[None]
        penalty = np.linalg.norm(distances, axis=-1).min(axis=0).mean()  # Number of wheels off track
        reward -= self.penalty_coeff * penalty
        self.reward -= self.penalty_coeff * penalty
        return obs, reward, done, truncated, info


gym.register(id="CarRacingPenalty-v0", entry_point=CarRacingPenaltyEnv)
