from jax_rl_util.baselines.brax_baselines import load_brax_model


def load_brax_baseline_inference_fn(env_name: str, backend: str, obs_size, act_size):
    """Load a trained Brax baseline policy function.

    Args:
        env_name (str): The name of the environment.
        obs_size (int): The size of the observation space.
        act_size (int): The size of the action space.
        backend (str): The name of the brax backend.

    Returns:
        A pretrained policy function for the given brax environment and backend.
    """
    import os

    file_dir = os.path.dirname(os.path.abspath(__file__))
    path = file_dir + f"/trained/brax_baselines/{backend}/{env_name}.ckpt"
    return load_brax_model(
        path,
        env_name=env_name,
        obs_size=obs_size,
        act_size=act_size,
    )
