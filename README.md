# jax-rl-util

Reusable reinforcement learning utilities for JAX, supporting online and offline RL across a range of environments and model types.

> Work in Progress

## Modules

### `envs`
Environment wrappers and utilities for Brax, Gymnasium, gymnax, and PopJym. Includes `VmapWrapper` for batched rollouts, `RandomizedAutoResetWrapper`, and helpers for rendering and reward aggregation.

### `td_lambda`
Online TD(λ) learning primitives:
- `trace_update` — accumulate or dutch eligibility trace update (`z ← γλz + ∇f`)
- `compute_td_updates` — multiply trace by TD-error to produce parameter updates
- `init_trace` — initialise a zero trace matching a parameter tree
- `RNNActorCritic` — `nn.RNNCellBase` combining an optional shared `RNNEnsemble` backbone with a `PolicyRNN` actor and an `RNNEnsemble` critic, suitable for real-time recurrent RL
- `Actor` — alias for `DistributionLayer` (from `jax_rtrl`)
- `Critic` — `DistributionLayer` subclass with `out_size=1` and `Deterministic` distribution

### `baselines`
PPO implementations: standard MLP (`ppo_agent`), recurrent (`ppo_rnn`), S5-based (`ppo_s5`), and environment-specific variants for Brax, BSuite, and highway-env.

### `optimizers`
`OptimizerConfig` dataclass and `make_optimizer` factory wrapping optax, with support for cosine/linear LR decay, gradient clipping, weight decay, and per-parameter group transforms via `make_multi_transform`.

### `eval` / `util`
Logging utilities (`DummyLogger`, `with_logger`, `leaf_norms`, `tree_norm`), checkpointing helpers, and misc JAX utilities.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, JAX < 0.7, and brax ≥ 0.12.

## Citation

If you use this library in your research, please consider citing:

```bibtex
@article{lemmel2024,
  title  = {Real-Time Recurrent Reinforcement Learning},
  author = {Lemmel, Julian and Grosu, Radu},
  year   = {2024},
  month  = mar,
  url    = {http://arxiv.org/abs/2311.04830},
  doi    = {10.48550/arXiv.2311.04830},
}
```
