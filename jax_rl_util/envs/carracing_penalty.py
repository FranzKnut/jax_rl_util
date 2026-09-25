"""A version of CarRacing environment with penalty for going off track."""

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import (
    CarRacing,
    ZOOM,
    SCALE,
    WINDOW_H,
    WINDOW_W,
    VIDEO_W,
    VIDEO_H,
    STATE_W,
    STATE_H,
)
import numpy as np
import pygame


class CarRacingPenaltyEnv(CarRacing):
    """A version of CarRacing environment with penalty for going off track and additional reward for going fast."""

    def __init__(
        self,
        penalty_coeff: float = 0.01,
        speed_reward_coeff: float = 0.0,
        speed_reward_distance_boundary: float = 1.0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        penalty_coeff : float, optional
            The coefficient for the penalty for going off track, by default 0.01
        speed_reward_coeff : float, optional
            The coefficient for the reward for going fast, by default 0.0
        speed_reward_distance_boundary : float, optional
            The distance boundary for the speed reward, no reward is given beyond this distance, by default 1.0
        """
        super().__init__(**kwargs)
        self.penalty_coeff = penalty_coeff
        self.speed_reward_coeff = speed_reward_coeff
        self.speed_reward_distance_boundary = speed_reward_distance_boundary

    def _get_distance_to_track_centerline(self):
        """Compute the distance to the track centerline."""
        position = np.array(self.car.hull.position)
        distances = np.array(self.track)[:, -2:] - position
        dist_to_closest = np.linalg.norm(
            distances, axis=-1
        ).min()  # distance to track centerline
        return dist_to_closest

    def _get_speed(self):
        """Compute the speed of the car."""
        speed = np.linalg.norm(np.array(self.car.hull.linearVelocity))
        return speed

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        _, reward_info = self.compute_reward()
        info["pos"] = np.array(self.car.hull.position)
        info.update(reward_info)
        info["base_reward"] = 0.0  # No base reward on reset
        return obs, info

    def compute_reward(self):
        """Compute the reward for the current step."""
        distance = self._get_distance_to_track_centerline()
        reward = -self.penalty_coeff * np.log(distance + 1)

        speed = self._get_speed()
        reward += self.speed_reward_coeff * speed * np.exp(-distance)
        return reward, {"penalty": distance, "speed": speed}

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Apply penalty for going off track
        extra_reward, reward_info = self.compute_reward()
        reward_info["base_reward"] = reward
        reward += extra_reward

        info["pos"] = np.array(self.car.hull.position)
        info.update(reward_info)
        return obs, reward, done, truncated, info

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"] + ["background"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        if mode == "background":
            zoom = 2.0
            trans = (WINDOW_W / 2, WINDOW_H / 2)
            angle = 0

        self._render_road(zoom, trans, angle)

        # if mode != "background":
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode != "background":
            # showing stats
            self._render_indicators(WINDOW_W, WINDOW_H)

            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "background":
            return self._create_image_array(self.surf, (WINDOW_W, WINDOW_H))
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen


gym.register(id="CarRacingPenalty-v0", entry_point=CarRacingPenaltyEnv)
