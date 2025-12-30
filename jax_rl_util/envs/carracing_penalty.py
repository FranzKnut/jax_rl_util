"""A version of CarRacing environment with penalty for going off track."""

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing


class CarRacingPenaltyEnv(CarRacing):
    """A version of CarRacing environment with penalty for going off track."""

    def __init__(self, penalty_coeff: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.penalty_coeff = penalty_coeff

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Apply penalty for going off track
        tile_contacts = [len(w.tiles) >= 1 for w in self.car.wheels]
        penalty = 4 - sum(tile_contacts)  # Number of wheels off track
        reward -= self.penalty_coeff * penalty
        self.reward -= self.penalty_coeff * penalty
        return obs, reward, done, truncated, info


gym.register(id="CarRacingPenalty-v0", entry_point=CarRacingPenaltyEnv)
