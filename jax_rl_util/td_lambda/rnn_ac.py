"""RNN Actor-Critic flax Module.

Moved to `jax_rtrl.networks.actor_critic`, so that every trainer shares one model and checkpoint
layout. Kept here so existing imports keep working.
"""

from jax_rtrl.networks.actor_critic import Actor, Critic, RNNActorCritic

__all__ = ["Actor", "Critic", "RNNActorCritic"]
