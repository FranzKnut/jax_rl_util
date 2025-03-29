"""RNNs built with Haiku."""
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrandom


class BaseRNNCell(hk.Module):
    """Base class for RNN cells built with equinox."""

    def map_to_y0(self, i, h):
        """Concatenate input and hidden state to form y0 for the ODE solver."""
        return jnp.concatenate([i, h], axis=-1)

    def __call__(self, inputs, hx, evolving_out=False, validation=False):
        """Compute the output of the cell.

        Args:
            inputs (_type_): input array
            hx (_type_): hidden state
            evolving_out (bool, optional): Return the whole solver sequence. Defaults to False.
            validation (bool, optional): Disable training behavior. Defaults to False.

        Returns:
            _type_: Output of the cell
        """
        raise NotImplementedError

    def get_initial_state(self, batch_size=1):
        """Get the initial state of the cell given batchsize."""
        return jnp.tile(self.h0, (batch_size, 1)) if batch_size else self.h0


class BaseODECell(BaseRNNCell):
    """Base class for ODE cells built with equinox."""

    t1 = 1.0
    dt = 1.0
    units = 16

    def __call__(self, hx, inputs, evolving_out=False, validation=False):
        """Solve the ODE and return the output."""
        y0 = self.map_to_y0(inputs, hx)
        solution = self.solve(y0, self.t1, self.dt, validation)

        if isinstance(solution, tuple):
            if evolving_out:
                out = solution[0][:, -self.units:]
                traces = solution[1]
            else:
                out = solution[0][-1, -self.units:]
                traces = jax.tree_map(lambda x: x[-1], solution[1])
            return (out[-self.units:], traces)
        else:
            out = solution[:, -self.units:]
            return out

    def f(self, t, y):
        """Differential equation to solve."""
        raise NotImplementedError

    def solve_euler(self, y0, t1, dt, validation=False):
        """Solve euler's method for the ODE specified by f.

        @param inputs:
        @param hidden_state:
        @return:
        """

        def euler_step(y, t):
            dh = self.f(t, y)
            out = jax.tree_map(lambda a, b: a + b * dt, y, dh)
            return out, out

        t = jnp.arange(t1 // dt)
        return hk.scan(euler_step, y0, t)

    def solve(self, y0, t1, dt, validation=False):
        """Solver wrapper.

        Args:
            y0 (_type_): initial state
            t1 (_type_): end time
            dt (_type_): time delta
            validation (bool, optional): Unused. Defaults to False.

        Returns:
            _type_: _description_
        """
        solution, ctx = self.solve_euler(y0, t1, dt, validation)
        return solution


class CTRNNCell_simple_hk(BaseODECell):
    """A simpler CTRNN cell with no affine transformation in synapses.

    This implementation follows (Murray 2019). https://elifesciences.org/articles/43299
    """

    def __init__(self, input_size, units, tau_min=1, tau_max=10,
                 weight_init_gain=1, key=None, **kwargs):
        """Initialize a simple CTRNN cell.

        Args:
            input_size (_type_): _description_
            units (_type_): _description_
            activation (_type_, optional): _description_. Defaults to jax.nn.tanh.
            tau_min (int, optional): _description_. Defaults to 1.
            tau_max (int, optional): _description_. Defaults to 10.
            weight_init_gain (int, optional): _description_. Defaults to 1.
            key (_type_, optional): _description_. Defaults to None.
        """
        super(CTRNNCell_simple_hk, self).__init__()
        self.in_size = input_size  # bias
        self.units = units
        self.keys = jrandom.split(key)
        # self.mask = jnp.concatenate([self.mask, jnp.zeros((all_post, 1))], axis=-1)

    def f(self, t, y):
        """CTRNN update."""
        w_shape = (self.units, self.in_size + self.units + 1)  # +1 for bias
        w = hk.get_parameter('w', w_shape, init=hk.initializers.RandomNormal(1.0))
        tau = hk.get_parameter('tau', [self.units], init=hk.initializers.RandomUniform(1, 10))
        x = y[:, -self.units:]
        # add one to each batch of y to account for bias
        _y = jnp.concatenate([y, jnp.ones(y.shape[:-1] + (1,))], axis=-1)
        u = jnp.einsum('ij,...j->...i', w, _y)
        act = jnp.tanh(u)
        dh = (act - x) / tau
        return jnp.concatenate([jnp.zeros((1, self.in_size)), dh], axis=-1)
