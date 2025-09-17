"""RL utilities module."""

from contextlib import contextmanager
import os

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


def try_init_metal():
    try:
        # Figure out if we are on Apple Silicon
        if "arm64" in os.popen("uname -m").read() and not os.system(
            "ioreg -l | grep gpu-core-count"
        ):
            # Activate METAL platform
            jax.config.update("jax_platforms", "cpu,METAL")
            jax.devices("METAL")
    except Exception as e:
        print(f"Failed to initialize METAL: {e}")
    # finally:
    #     # Set default device to CPU
    #     jax.config.update("jax_default_device", jax.devices("cpu")[0])
