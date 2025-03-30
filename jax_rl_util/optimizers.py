"""Util for creating optax optimizers."""

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from optax._src import base, wrappers


@dataclass(frozen=True)
class OptimizerConfig:
    """Class representing the parameters for an optimizer."""

    # fmt: off
    opt_name: str = "adam"                                       # opt_name (str): The name of the optimizer.
    learning_rate: float = 1e-3                                 # learning_rate (float): The learning rate for the optimizer.
    kwargs: dict = field(default_factory=dict, hash=False)      # kwargs (dict): Additional keyword arguments for the optimizer.
    lr_decay_type: str | None = None                            # decay_type (str): The type of decay for the learning rate.
    lr_kwargs: dict = field(default_factory=dict, hash=False)   # lr_kwargs (dict): Additional keyword arguments for the learning rate decay.
    reg_type: str | None = None                                 # reg_type (str): The type of weight regularization.
    weight_decay: float = 0.0                                   # weight_decay (float): The strenght of weight regularization.
    gradient_clip: float | None = None                          # gradient_clip (float): The value to clip the gradients.
    multi_step: int | None = None                               # multi_step (int): number of steps to accumulate.
    reduce_on_plateau: bool = False                             # reduce_on_plateau (bool): Reduce learning rate on plateau.
    # fmt: on


def make_optimizer(config=OptimizerConfig(), direction="min") -> optax.GradientTransformation:
    """Make optax optimizer.

    The decorator allows reading scheduled lr from the optimizer state.

    Parameters
    ----------
    config : OptimizerConfig, optional
        learning_rate : float
            initial learning rate
        direction : str, optional
            min or max. Defaults to "min", by default "min"
        opt_name : str, optional
            Name of optimizer, by default 'sgd'
        gradient_clip : int, optional
            Clip gradient norm. Defaults to 0
        lr_decay : int, optional
            Exponential lr decay. Defaults to 1, by default 1
        optimizer_params : dict, optional
            Additional kwargs to the optimizer, by default {}
    direction : str, optional
        min or max. Defaults to "min".

    Returns
    -------
        optax optimizer
    """
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    if direction in ["max", "maximize"]:
        learning_rate = -learning_rate
    else:
        weight_decay = -weight_decay

    if config.lr_decay_type == "cosine_warmup":
        """Args:
            initial_multiplier: Scalar multiplier for the initial learning rate.
            end_multiplier: Scalar multiplier for the final learning rate.
            warmup_steps: Positive integer, the length of the linear warmup.
            decay_steps: Positive integer, the total length of the schedule. Note that
                this includes the warmup time, so the number of steps during which cosine
                annealing is applied is ``decay_steps - warmup_steps``.
        """
        learning_rate = optax.warmup_cosine_decay_schedule(
            learning_rate * config.lr_kwargs.get("initial_multiplier", 0),
            peak_value=learning_rate,
            end_value=learning_rate * config.lr_kwargs["end_multiplier"],
            decay_steps=config.lr_kwargs["decay_steps"],
            warmup_steps=config.lr_kwargs["warmup_steps"],
        )

    elif config.lr_decay_type == "warmup":
        """Schedule with linear transition from ``init_value`` to ``end_value``.

        Args:
            init_value: initial value for the scalar to be annealed.
            end_value: end value of the scalar to be annealed.
            transition_steps: number of steps over which annealing takes place. The
                scalar starts changing at ``transition_begin`` steps and completes the
                transition by ``transition_begin + transition_steps`` steps. If
                ``transition_steps <= 0``, then the entire annealing process is disabled
                and the value is held fixed at ``init_value``.
            transition_begin: must be positive. After how many steps to start annealing
                (before this many steps the scalar value is held fixed at ``init_value``).

        Returns:
            schedule
            A function that maps step counts to values.
        """
        learning_rate = optax.linear_schedule(
            init_value=learning_rate * config.lr_kwargs["initial_multiplier"],
            end_value=learning_rate,
            transition_steps=config.lr_kwargs["warmup_steps"],
        )
    elif config.lr_decay_type == "cosine":
        """Args:
            init_value: An initial value for the learning rate.
            decay_steps: Positive integer - the number of steps for which to apply
                the decay for.
            alpha: The minimum value of the multiplier used to adjust the
                learning rate. Defaults to 0.0.
            exponent:  The default decay is ``0.5 * (1 + cos(pi * t/T))``, where 
                ``t`` is the current timestep and ``T`` is the ``decay_steps``. The
                exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``.
                Defaults to 1.0.

        """
        learning_rate = optax.cosine_decay_schedule(
            learning_rate, decay_steps=config.lr_kwargs["decay_steps"], alpha=config.lr_kwargs.get("alpha", 0)
        )
    elif config.lr_decay_type == "exponential":
        """Args:
            init_value: the initial learning rate.
            transition_steps: must be positive. See optax docs for decay computation.
            decay_rate: must not be zero. The decay rate.
            transition_begin: must be positive. After how many steps to start annealing
                (before this many steps the scalar value is held fixed at `init_value`).
            staircase: if `True`, decay the values at discrete intervals.
            end_value: the value at which the exponential decay stops. When
                `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
                an upper bound. Has no effect when `decay_rate` = 0.
        """
        learning_rate = optax.exponential_decay(
            learning_rate,
            config.lr_kwargs["transition_steps"],
            config.lr_kwargs["decay_rate"],
            config.lr_kwargs.get("warmup_steps", 0),
            config.lr_kwargs.get("staircase", False),
            config.lr_kwargs.get("end_value", None),
        )
    elif config.lr_decay_type is not None:
        raise ValueError(f"Decay type {config.lr_decay_type} unknown.")

    if weight_decay == "l2" and config.opt_name in ["adam"]:
        print(f"WARNING: Weight decay incorrect for {config.opt_name}, consider using adamw.")

    @optax.inject_hyperparams
    def _make_opt(learning_rate):
        _opt = getattr(optax, config.opt_name)

        reg_types = {
            "l2": optax.add_decayed_weights(weight_decay),
            "l1": add_decayed_weights_l1(weight_decay),
        }

        # Create optimizer from optax chain
        optimizer = optax.chain(
            # Weight decay
            reg_types.get(config.reg_type, optax.identity())
            if config.opt_name not in ["adamw"]
            else optax.identity(),  # , mask=decay_mask
            # Gradient clipping
            optax.clip_by_global_norm(config.gradient_clip) if config.gradient_clip else optax.identity(),
            # Optimizer
            _opt(learning_rate, **config.kwargs)
            if config.opt_name not in ["adamw"]
            else _opt(learning_rate, **config.kwargs, weight_decay=weight_decay),
            # Reduce on Plateau
            optax.contrib.reduce_on_plateau(
                patience=config.lr_kwargs.get("patience", 100),
                factor=config.lr_kwargs.get("factor", 0.5),
                min_scale=config.lr_kwargs.get("min_scale", 1e-6),
                accumulation_size=config.lr_kwargs.get("accumulation_size", 10),
            )
            if config.reduce_on_plateau
            else optax.identity(),
        )
        if config.multi_step:
            optimizer = optax.MultiSteps(optimizer, every_k_schedule=config.multi_step)
        return optimizer

    return _make_opt(learning_rate)


def add_decayed_weights_l1(
    weight_decay: Union[float, jax.Array] = 0.0, mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
    """Add the derivative of L1 loss of the params scaled by `weight_decay`.

    More precisely, this will add sign(p) for each entry p.

    Args:
      weight_decay: A scalar weight decay rate.
      mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
      A `GradientTransformation` object.
    """

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        additions = jtu.tree_map(lambda p: jnp.sign(p), params)
        updates = jtu.tree_map(lambda g, p: g + weight_decay * p, updates, additions)
        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return wrappers.masked(base.GradientTransformation(base.init_empty_state, update_fn), mask)
    return base.GradientTransformation(base.init_empty_state, update_fn)


def label_subtrees(params, subtrees):
    """Make Prefix subtree.

    Parameters
    ----------
    params : tree
        A nested dict of parameters.
    subtrees : list of subtree names
        List of subtree names to replace.

    Returns
    -------
    tree
        A subtree with the same structure as params,
        but with the name matching subtrees replaced by their name.
    """
    for k, v in params.items():
        if k in subtrees:
            return k
        else:
            return label_subtrees(v, subtrees)


def make_multi_transform(configs: dict, label_fn: callable = None):
    """Make optax multi_transform for given (nested) dict of configs.

    keys in configs should match subtrees in params.

    Parameters
    ----------
    configs : dict
        A nested dict of optimizer configs.
    label_fn : callable, optional
        A function that labels subtrees of the model parameter dict with keys of the configs dict.

    Returns
    -------
        optax optimizer
    """
    optimizers = {k: make_optimizer(v) for k, v in configs.items()}
    label_fn = label_fn or partial(label_subtrees, subtrees=list(configs.keys()))
    return optax.multi_transform(optimizers, label_fn)


def get_current_lrs(opt_state, opt_config: OptimizerConfig | None = None):
    """Get current learning rate from optimizer state."""
    lrs = {}
    _reduce_on_plateau = False if opt_config is None else opt_config.reduce_on_plateau
    if hasattr(opt_state, "inner_states"):
        for k, s in opt_state.inner_states.items():
            reduce_on_plateau_lr = s[0][3][3].scale if _reduce_on_plateau else 1
            lrs["LR/" + k] = s[0][1]["learning_rate"] * reduce_on_plateau_lr
    else:
        reduce_on_plateau_lr = opt_state[3][3].scale if _reduce_on_plateau else 1
        lrs["learning_rate"] = opt_state[1]["learning_rate"] * reduce_on_plateau_lr
    return lrs


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict / pytree.

    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn


def make_optimizer_for_model(model_name: str, config=OptimizerConfig(), no_decay_lr_factor=1.0):
    """Make optax optimizer for given model name and config."""
    if "s5" in model_name:
        no_decay_params = ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
    elif "lru" in model_name:
        no_decay_params = ["B_re", "B_im", "nu_log", "theta_log", "gamma_log"]
    elif model_name in ["ctrnn", "rflo", "bptt"]:
        no_decay_params = ["W", "tau"]
    else:
        return make_optimizer(config)

    print("Making optimizer for", model_name, "model, no_decay_params:", no_decay_params)
    ssm_fn = map_nested_fn(lambda k, _: "no_decay" if k in no_decay_params else ("none" if k in [] else "regular"))
    return make_multi_transform(
        {
            "none": replace(config, learning_rate=0.0),
            "no_decay": replace(
                config, opt_name="adam", learning_rate=no_decay_lr_factor * config.learning_rate, weight_decay=0.0
            ),
            "regular": config,
        },
        ssm_fn,
    )
