"""Triangular swimmer in two dimensions that is propelling itself by applying force to its arms."""

from typing import Any, Dict, Optional, Tuple, Union

import chex
import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax import struct

# jax.config.update("jax_enable_x64", True)


"""
position of the beads - ((x1,y1),(x2,y2),(x3,y3)).
OBSERVATION - [L12,L23, L13] or [ab, bc, ca] - arm lengths of the swimmer.
output - [f12, f23, f13] or [f_{ab}, f_{bc}, f_{ca}] - active forces on the arms.

individual instanteneous rewards are smaller. 
Cumulative rewards are large as they are summed over individual small rewards. 
We need to increase individual rewards. 
In NEAT, the notion is to increase the value of individual rewards. 
that is why it works there I guess. We have to find a way 
to increase the individual rewards here. Problem will be solved

Created by Ruma Maity
Adjusted by jlemmel
"""


@struct.dataclass
class TriState(gymnax.environments.environment.EnvState):
    init_pos: jnp.ndarray
    pos: jnp.ndarray
    mobility: jnp.ndarray
    _state_b: jnp.ndarray
    _velocity_b: jnp.ndarray
    _state_com: jnp.ndarray
    _velocity_com: jnp.ndarray
    _angular_velocity_b: jnp.ndarray
    _eq_length: jnp.ndarray
    _angular_velocity: jnp.ndarray


@struct.dataclass
class EnvParams(gymnax.environments.environment.EnvParams):
    eta_inv: float = 1 / (8 * jnp.pi * 1)  # 1.0 / (8 * jnp.pi * viscosity)
    # force_scale: float = 1.0
    # TODO: Add all the params from __init__


class TriangleJax(gymnax.environments.environment.Environment):
    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=5000)

    def __init__(self, **env_kwargs):
        super().__init__()
        self.final_time = 5000  # how long the simulation will run
        self.dt = 1
        self.beads = ["a", "b", "c"]
        self.linkers = ["ab", "bc", "ca"]
        self.num_beads = len(self.beads)
        self.dimension = 2
        self.length_scale = 10  # 1
        self.radius = 0.1 * self.length_scale  # 1
        self.force_scale = env_kwargs.get("force_scale", 10.0)
        self.k = 10.0
        self.reward_weights = jnp.array(env_kwargs.get("reward_weights", [0.75, 0.25, 0, 0]))

        self.length_range = jnp.array([1.0, 1.4]) * self.length_scale  # permissible range of the arms
        self.lengths = jnp.zeros(3)
        self.mobility = jnp.zeros((self.num_beads, self.num_beads, self.dimension, self.dimension))

        self.positions_to_displacements, self.displacements_to_forces = self._make_matrices()

    def observation_space(self, params):
        return gymnax.environments.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

    def action_space(self, params):
        return gymnax.environments.spaces.Box(low=-jnp.ones(3), high=jnp.ones(3), shape=(3,))

    def reset_env(self, key, params: EnvParams) -> Tuple[chex.Array, gymnax.EnvState]:
        self.mobility = jnp.zeros((self.num_beads, self.num_beads, self.dimension, self.dimension))
        mu = ((4.0 / 3.0) * params.eta_inv / self.radius) * jnp.ones((3,))

        for i in range(self.num_beads):
            self.mobility = self.mobility.at[i, i].set(mu[i] * jnp.eye(self.dimension))
        angle = jrandom.uniform(key, (3,), minval=0, maxval=0.1)  # shape 3
        lscale = self.length_scale
        pos = (
            jnp.array(
                [[-0.5, -0.29 + angle[0] / lscale], [0.5, -0.29 + angle[2] / lscale], [0 + angle[1] / lscale, 0.57]]
            ).reshape((3, 2))
            * self.length_scale
        )
        init_state_com = jnp.mean(pos, axis=0)
        init_pos = init_state_com
        initial_relative_posb = pos - init_state_com  # self._init_state
        _eq_length = (
            jnp.linalg.norm(initial_relative_posb[0,] - initial_relative_posb[1])
            + jnp.linalg.norm(initial_relative_posb[1] - initial_relative_posb[2])
            + jnp.linalg.norm(initial_relative_posb[0] - initial_relative_posb[2])
        ) / 3

        pos_a = pos[0, :]
        pos_b = pos[1, :]
        pos_c = pos[2, :]
        arm_ab = jnp.linalg.norm(pos_a - pos_b)
        arm_bc = jnp.linalg.norm(pos_c - pos_b)
        arm_ca = jnp.linalg.norm(pos_a - pos_c)
        state_p = jnp.array([arm_ab, arm_bc, arm_ca]).reshape((3,))  # arm-length, velocity
        state_v = jnp.zeros(self.dimension)
        state = TriState(
            0,
            pos,
            pos,
            _state_b=initial_relative_posb,
            mobility=self.mobility,
            _eq_length=_eq_length,
            _velocity_b=jnp.zeros((3, 2)),
            _state_com=init_state_com,
            _velocity_com=jnp.zeros((2,)),
            _angular_velocity_b=jnp.zeros((3)),
            _angular_velocity=jnp.zeros((1)),
        )
        return state_p, state  # jnp.concatenate([state_p, state_v]), state

    def get_rewards(self, state: TriState) -> float:
        pos = state.pos
        _state_com = jnp.mean(pos, axis=0)
        com_dis = _state_com - jnp.mean(state.init_pos, axis=0)
        x_rel = state.pos - _state_com
        rel_pos_norm = jnp.linalg.norm(x_rel, axis=1)
        ang_vel = jnp.sum(state._angular_velocity_b)
        min_arm_len = jnp.min(rel_pos_norm)
        # length_max = jnp.max(jnp.asarray(rel_pos_list)) / 4
        vel = jnp.linalg.norm(state._velocity_com)
        # vel = state._velocity_com[0]  # velocity in x direction
        reward1 = jnp.linalg.norm(com_dis) / (state.time + 1)  # / self.length_scale
        reward2 = jnp.abs(vel)  # / self.length_scale
        reward3 = jnp.abs(ang_vel)
        reward4 = min_arm_len  # / self.length_scale
        all_rewards = jnp.array([reward1, reward2, reward3, reward4])
        reward = jnp.dot(self.reward_weights, all_rewards)
        return jnp.array([reward, reward1, reward2, reward3, reward4])

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TriState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, TriState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        if params is None:
            params = self.default_params
        action *= self.force_scale
        action = jnp.clip(action, -self.force_scale, self.force_scale)
        prev_state = state
        state = self.simulate(state, action, params)
        reward = self.get_rewards(state)
        done = False

        def calc_lengths(pos):
            pos_a = pos[0, :]
            pos_b = pos[1, :]
            pos_c = pos[2, :]
            arm_ab = jnp.linalg.norm(pos_a - pos_b)
            arm_bc = jnp.linalg.norm(pos_c - pos_b)
            arm_ca = jnp.linalg.norm(pos_a - pos_c)
            return jnp.array([arm_ab, arm_bc, arm_ca]).reshape((3,))

        state_p = calc_lengths(state.pos)
        state_v = state._velocity_com
        done = state.time >= params.max_steps_in_episode / self.dt
        return (
            state_p,
            state,
            reward[0],
            done,
            {"state": state.pos},
        )  # jnp.concatenate([state_p, state_v]), state, reward[0], done, {"state": state.pos}

    # @staticmethod
    def _make_matrices(self):
        nb = 3  # number of beads
        nl = 3  # number of linkers or arms
        p_to_d = jnp.zeros([nl, nb])
        d_to_f = jnp.zeros([nb, nl])
        dimension = 2
        for i, l in enumerate(self.linkers):
            b0, b1 = l[0], l[1]
            a, b = self.beads.index(b0), self.beads.index(b1)
            p_to_d = p_to_d.at[i, a].set(-1)
            p_to_d = p_to_d.at[i, b].set(1)
            d_to_f = d_to_f.at[a, i].set(-1 * self.k * self.force_scale / self.length_scale)
            d_to_f = d_to_f.at[b, i].set(1 * self.k * self.force_scale / self.length_scale)
        positions_to_displacements = p_to_d[:, :, jnp.newaxis, jnp.newaxis] * jnp.eye(dimension)
        displacements_to_forces = d_to_f[:, :, jnp.newaxis, jnp.newaxis] * jnp.eye(dimension)
        return positions_to_displacements, displacements_to_forces

    def simulate(self, state: TriState, action, params: EnvParams) -> TriState:
        act = action.reshape((3, 1))
        comp_a = act * compute_cos_sin(state.pos)

        final_position, new_mobility = midpoint_integrate(
            x_0=state.pos,
            t=self.dt,
            n=30,
            a=comp_a,
            k=self.k * self.force_scale / self.length_scale,
            lr=jnp.array([0.8, 1.2]) * self.length_scale,
            p_to_d=self.positions_to_displacements,
            d_to_f=self.displacements_to_forces,
            mobility=state.mobility,
            eta_inv=params.eta_inv,
        )
        dx = final_position - state.pos
        d_com = jnp.array([0.5, 0.5, 0.5]) @ dx / 3
        d_rel = dx - d_com
        new_state_com = state._state_com + d_com
        new_state_b = state._state_b + d_rel
        new_pos = new_state_com + new_state_b
        r = state._eq_length
        rel_vel = d_rel / self.dt
        new_angular_velocity_b = (new_state_b[:, 0] * rel_vel[:, 1] - new_state_b[:, 1] * rel_vel[:, 0]) / r**2

        return state.replace(
            time=state.time + 1,
            pos=new_pos,
            mobility=new_mobility,
            _state_b=new_state_b,
            _velocity_b=rel_vel,
            _state_com=new_state_com,
            _velocity_com=d_com / self.dt,
            _angular_velocity_b=new_angular_velocity_b,
        )


def update_mobility_jax(x, mobility, eta_inv):
    n = mobility.shape[0]
    for a in range(n):
        for b in range(a):
            d = x[b] - x[a]
            l_inv = 1 / jnp.linalg.norm(d)
            d_outer = jnp.outer(d, d) * l_inv**2
            diag = jnp.eye(x.shape[1]) * eta_inv * l_inv
            off_diag = eta_inv * l_inv * d_outer
            mobility = mobility.at[b, a].set(diag + off_diag)
            mobility = mobility.at[a, b].set(diag + off_diag)
    return mobility


def midpoint_integrate(x_0, t, n, a, k, lr, p_to_d, d_to_f, mobility, eta_inv):
    step = t / n

    def _step(_, carry):
        x, mobility = carry
        f1 = calculate_forces(x, a, k, lr, p_to_d, d_to_f)  # *10 # force scale
        mobility = update_mobility_jax(x, mobility, eta_inv)
        v1 = four_two_contraction(mobility, f1)
        x_inner = x + v1 * step * 0.5

        f2 = calculate_forces(x_inner, a, k, lr, p_to_d, d_to_f)  # *10 # force scale
        mobility = update_mobility_jax(x_inner, mobility, eta_inv)
        v2 = four_two_contraction(mobility, f2)
        x += v2 * step
        return x, mobility

    return jax.lax.fori_loop(0, n, _step, (x_0, mobility))


def compute_cos_sin(pos):
    edges = jnp.array([pos[0] - pos[1], pos[1] - pos[2], pos[2] - pos[0]])
    edge_lengths = jnp.linalg.norm(edges, axis=1)
    edge_lengths = jnp.where(edge_lengths == 0, 1, edge_lengths)
    return edges / edge_lengths[:, None]


def calculate_forces(x, a, k, lr, p_to_d, d_to_f):
    d = four_two_contraction(p_to_d, x)
    _l = jnp.linalg.norm(d, axis=1)
    l_0 = jnp.clip(_l, min=lr[0], max=lr[1])

    cos_sin = compute_cos_sin(x)
    comp_l_0 = l_0[:, None] * cos_sin
    comp_l_c = _l[:, None] * cos_sin
    comp_l_c = jnp.where(comp_l_c == 0, 1, comp_l_c)
    f = four_two_contraction(d_to_f, (d * (-1 + (a / k + comp_l_0) / comp_l_c)))
    return f


def four_two_contraction(M, f):
    return jnp.einsum("abij,bj->ai", M, f)
