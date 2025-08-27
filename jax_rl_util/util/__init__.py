"""RL utilities module."""

from contextlib import contextmanager

import jax


@contextmanager
def run_on_gpu():
    """Context manager to run code on GPU if available."""
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        # Also try metal for Apple Silicone
        try:
            device = jax.devices("METAL")[0]
        except RuntimeError:
            print("WARNING: No GPU available, using CPU for training.")
            device = None
    with jax.default_device(device):
        yield
