import json
import os

import dacite
import gymnasium
import jax
import jax.numpy as jnp
import jax.random as jrandom
import models.mlp as models
import numpy as np
import orbax.checkpoint
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers.frame_stack import FrameStack
from models.jax_util import JaxRng

from jax_rl_util.envs.wrappers import GymJaxWrapper, is_discrete
from jax_rl_util.envs.wrappers import RGBtoGrayWrapper
from rtr_iil.reinforcement.ppo_rnn import ActorCriticRNN, PPO_Params


def load_config(ckpt_dir):
    config_file = os.path.join(ckpt_dir, "config.json")
    with open(config_file, "r") as f:
        return dacite.from_dict(PPO_Params, json.load(f))


def load_params(ckpt_dir):
    ckpt_file = os.path.abspath(os.path.join(ckpt_dir, "best.pkl"))
    return orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_file)


class PPO_Agent(JaxRng):
    def __init__(self, action_dim, discrete, params, config, rng=None) -> None:
        rng = rng or jrandom.PRNGKey(0)
        super().__init__(rng)
        # Restore the model config and parameters
        self.params = params
        self.config = config

        # Initialize the model
        self.model = ActorCriticRNN(action_dim, self.config, discrete)
        rnn_cls = getattr(models, self.config.MODEL)

        self.hidden = rnn_cls.initialize_carry(1, self.config.NUM_UNITS)

    def __call__(self, obs, done):
        self.hidden, pi, value = jax.jit(self.model.apply)(
            self.params, self.hidden, (obs[None, None], jnp.array(done)[None, None])
        )
        return pi.sample(seed=self.rng).squeeze(), value


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    run_id = "44438aaac92d4c729a38e956"
    video_length = 10000
    ckpt_dir = os.path.join("artifacts", "aim", run_id)
    config = load_config(ckpt_dir)
    params = load_params(ckpt_dir)

    env = gymnasium.make(
        "CarRacing-v2", render_mode="rgb_array", **config.env_init_args
    )

    env = RecordVideo(env, video_folder="./artifacts/videos", video_length=video_length)
    if config.stack_frames > 1:
        env = FrameStack(env, config.stack_frames)

    env = GymJaxWrapper(env)
    env = RGBtoGrayWrapper(env)

    agent = PPO_Agent(env.action_size, is_discrete(env), params, config)

    done = False
    obs, _ = env.reset(None)
    frames = []
    i = video_length

    while not done and i > 0:
        # obs = np.array(Image.fromarray(obs, mode='RGB').convert('L'))
        action, value = agent(obs[None], done)
        obs, state, reward, done = env.step(0, np.array(action), key=0)
        i -= 1
    env.close()
