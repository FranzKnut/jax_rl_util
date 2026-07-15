"""Environment Module."""

from jax_rl_util.util.config_util import dict_field


from dataclasses import dataclass, field


BRAX_ENVS_POS_DIMS = {"ant": 2, "halfcheetah": 1, "humanoid": 2}


@dataclass(frozen=True, eq=True)
class EnvironmentConfig:
    """Parameters for gym environments.

    Attributes
    ----------
        env_name (str): Environment name. Supported are brax, gymnax, popjym, highway_env, mujoco_playgound and gym envs.
        obs_mask (Union[str, tuple[int]]): Mask for the observation space.
        init_kwargs (dict): Initialization arguments for the environment.
        step_kwargs (dict): Arguments for the env step function.
        max_ep_length (int): Maximum episode length.
        batch_size (int): Number of parallel environments.
        transform_wrappers (list): List of wrappers to apply to the environment.
    """

    env_name: str = "CartPole-v1"
    # reward_scaling: int = 1
    obs_mask: str | tuple[int] | None = None
    init_kwargs: dict = dict_field(default={}, hash=False)
    step_kwargs: dict = dict_field(default={}, hash=False)
    max_ep_length: int = 1000
    batch_size: int | None = None
    transform_wrappers: list = field(default_factory=list, hash=False)

    @property
    def env_kwargs(self):
        """For backwards compatibility."""
        return self.step_kwargs


def print_env_info(env_info):
    """Print infos about an environment. Takes env_info from make_env."""
    env_name, package, OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = (
        env_info.values()
    )
    print(f"ENV:         {env_name}")
    print(f"package:     {package}")
    print(f"obs_size:    {OBS_SIZE}")
    print(f"act_size:    {ACT_SIZE}" + (" (discrete)" if DISCRETE else " (continuous)"))
    print(f"obs_mask:    {obs_mask}")
    print(f"act_clip:    {act_clip}")
    # print(f'value_size: {VALUE_SIZE}')
    
