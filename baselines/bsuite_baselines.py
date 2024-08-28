"""Train controller using bsuite built-in algorithms.

Note that it might be necessary to install additional dependencies to run this
"""

import os
import bsuite
import haiku as hk
from bsuite.logging import csv_logging
from bsuite.baselines import base
from bsuite.baselines import experiment
from bsuite.baselines.jax.actor_critic_rnn.agent import ActorCriticRNN
import optax
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from models.jax.rnns_haiku import CTRNNCell_simple_hk

EXPERIMENT = "MemoryLength"
MEMORY_LENGTHS = [4, 8, 16, 32, 64]
NUM_EPISODES = 1000
SEQ_LEN = 64
MODEL = 'CTRNN'
NUM_RUNS = 10
OUTPUT_PATH = f'./logs/bsuite/{MODEL}-{str(NUM_EPISODES)}EPS-{str(SEQ_LEN)}SEQ/'


def eval_agent(env, agent: base.Agent, steps=5000):
    """Evaluate a bsuite agent."""
    rewards = []
    dones = []
    for _ in range(steps):
        timestep = env.reset()
        while not timestep.last():
            action = agent.select_action(timestep)
            timestep = env.step(action)
            rewards.append(timestep.reward)
            dones.append(timestep.last())
        agent.select_action(timestep)

    total_reward = np.sum(sum(rewards) / max(sum(dones), 1))
    return np.mean(total_reward)


experiments = pd.DataFrame(index=[f"run-{i}" for i in range(NUM_RUNS)])


def make_agent(obs_spec, action_spec, seed: int = 0,
               model='LSTM', hidden_size=32, seq_len=32, td_lambda=0.99) -> base.Agent:
    """Create an actor-critic agent with default hyperparameters."""
    if model == 'LSTM':
        initial_rnn_state = hk.LSTMState(
            hidden=jnp.zeros((1, hidden_size), dtype=jnp.float32),
            cell=jnp.zeros((1, hidden_size), dtype=jnp.float32))

        def network(inputs: jnp.ndarray, state):
            flat_inputs = hk.Flatten()(inputs)
            torso = hk.nets.MLP([hidden_size, hidden_size])
            lstm = hk.LSTM(hidden_size)
            policy_head = hk.Linear(action_spec.num_values)
            value_head = hk.Linear(1)

            embedding = torso(flat_inputs)
            embedding, state = lstm(embedding, state)
            logits = policy_head(embedding)
            value = value_head(embedding)
            return (logits, jnp.squeeze(value, axis=-1)), state

    elif model == 'CTRNN':
        initial_rnn_state = jnp.zeros((1, hidden_size))

        def network(inputs, state):
            flat_inputs = hk.Flatten()(inputs)
            torso = CTRNNCell_simple_hk(inputs.shape[-1], hidden_size, key=jrandom.PRNGKey(seed))
            policy_head = hk.Linear(action_spec.num_values)
            value_head = hk.Linear(1)
            state = torso(state, flat_inputs)
            logits = policy_head(state)
            value = value_head(state)
            return (logits, jnp.squeeze(value, axis=-1)), state

    else:
        raise ValueError(f"Unknown model {model}")

    return ActorCriticRNN(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        initial_rnn_state=initial_rnn_state,
        optimizer=optax.adam(1e-2),
        rng=hk.PRNGSequence(seed),
        sequence_length=seq_len,
        discount=0.99,
        td_lambda=td_lambda,
    )


def train_multiple_for_length(length):
    """Train multiple agents for a given memory length."""
    eval_rewards = []
    print("Experiments for length {}".format(length))
    for i in range(NUM_RUNS):
        id = f'MemoryLength-{str(length)}-{str(i)}'
        experiments['fname'] = os.path.join(OUTPUT_PATH, f"bsuite_id_-_{id}.csv")
        env = bsuite.load("memory_len", {'memory_length': length})
        env = csv_logging.wrap_environment(env, bsuite_id=id, results_dir=OUTPUT_PATH, overwrite=True)
        agent = make_agent(
            obs_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            seed=i,
            model=MODEL,
            seq_len=SEQ_LEN,
        )
        print("    Training {}/{}".format(i + 1, NUM_RUNS))
        experiment.run(agent, env, num_episodes=NUM_EPISODES, verbose=True)
        print("    Evaluating")
        eval_rewards.append(eval_agent(env, agent))
    return eval_rewards


if EXPERIMENT == "MemoryLength":
    print("Model: {}".format(MODEL))
    for length in MEMORY_LENGTHS:
        experiments[length] = train_multiple_for_length(length)
        experiments.to_csv(os.path.join(OUTPUT_PATH, 'MemoryLength.csv'))
# elif EXPERIMENT == "AllGymnax":
#     print("Model: {}".format(MODEL))
#     for env in env_names:
#         experiments[length] = train_multiple_for_length(length)
#         experiments.to_csv(os.path.join(OUTPUT_PATH, 'MemoryLength.csv'))
