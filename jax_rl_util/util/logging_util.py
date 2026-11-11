"""Utilies for logging."""

import collections
import contextlib
import os
import traceback
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from operator import attrgetter
from typing import Literal

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import simple_parsing
from jax.tree_util import tree_reduce
from matplotlib import pyplot as plt
from PIL import Image
from typing_extensions import override


@dataclass
class LoggableConfig(simple_parsing.Serializable, decode_into_subclasses=True):
    """Base class for loggable configuration dataclasses."""

    logging: Literal["wandb", "aim", None] = None
    repo: str | None = None  # FIXME: misnomer! this is the wandb entity, not the repo
    project_name: str | None = None
    debug: bool | int = False
    log_code: bool = False


class DummyLogger(dict, object):
    """Dummy Logger that does nothing besides acting as dictionary."""

    run_id: str = "dummy"
    run_artifacts_dir: str = "artifacts/log"

    def __repr__(self) -> str:
        """Return name of logger."""
        return "DummyLogger"

    def flush(self):
        """Flush the logs."""
        pass

    def log(self, metrics: dict, step: int = None, **kwargs):
        """Log a dictionary of metrics (per step).

        Parameters
        ----------
        metrics : dict
            Dictonaries of scalar metrics.
        step : int, optional
            Step number, by default framework will use global step.
        kwargs : any
            Are passed to the underlying logging method.
        """
        pass

    def log_params(self, params_dict, **kwargs):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        pass

    def log_dist(self, values, step=None, **kwargs):
        """Log a distribution of values.

        Parameters
        ----------
        values : dict
            Dictonaries of values for distributions.
        step : int, optional
            Step number, by default framework will use global step.
        kwargs : any
            Are passed to the underlying logging method.
        """
        pass

    def finalize(self, all_param_norms=None, **kwargs):
        """Log additional plots or media.

        Parameters
        ----------
        all_param_norms : TODO
            _description_
        """
        pass

    def log_model(self, name: str, path: str, **kwargs):
        """Save a file as an artifact.
        TODO: make name optional and use the run hash as name if not provided.

        Parameters
        ----------
        name : str
            Name of the artifact.
        path : str
            Path to the file to be logged.
        """
        pass

    def log_img(self, name, img, step=None, caption="", pil_mode="RGB", **kwargs):
        """Log an image."""
        plt.imshow(img)
        plt.title(caption)
        out_file = (
            os.path.join(self.run_artifacts_dir, f"{name}_{step}.png")
            if step is not None
            else os.path.join(self.run_artifacts_dir, f"{name}.png")
        )
        os.makedirs(out_file.rsplit("/", 1)[0], exist_ok=True)
        plt.savefig(out_file)
        plt.close()
        print(f"DummyLogger: Saved image to {out_file}")

    def log_figure(self, name, fig, step=None, **kwargs):
        """Log a figure."""
        self.log({name: fig}, step=step)

    def log_video(self, name: str, frames, step: int | None = None, fps=4, **kwargs):
        """Save a video given as array.

        Parameters
        ----------
        name : str
            Name of the logged object.
        frames : array
            leading dimension for frames, then height, width, channels
        step : int, optional
            Step number, by default framework will use global step.
        fps : int, optional
            FPS for the video, by default 4
        kwargs : any
            Are passed to the underlying logging method.
        """
        file_name = f"{name}_{step}.gif" if step is not None else f"{name}.gif"
        save_video(frames, file_name, self.run_artifacts_dir, fps=fps)


def save_video(frames, file_name, out_dir, fps=30):
    """Save a video given as array.

    Parameters
    ----------
    frames : array
        leading dimension for frames, then height, width, channels
    file_name : str
        Name of the file to save the video to.
    """
    file_name = file_name.replace("/", "_")
    file_name = os.path.join(out_dir, file_name)
    num_frames = len(frames)
    print(f"Saving video to {file_name} with {num_frames} frames at {fps} fps.")
    images = [Image.fromarray(frames[i]) for i in range(num_frames)]
    os.makedirs(out_dir, exist_ok=True)
    images[0].save(
        file_name,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )


def update_nested_dict(d: dict, u: dict | None, path="", strict=True):
    """Update nested dict d with values from nested dict u.

    Parameters
    ----------
    d : dict
        Base dict
    u : dict
        Updates. If None, no updates are made and d is returned.

    Returns
    -------
    dict
        d with values overwritten by u
    """
    if u is None:
        return d
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(
                d.get(k) or {}, v, path + "." + str(k) if path else str(k), strict
            )
        else:
            if strict:
                assert k in d, f"Key {path + '.' + k} not in base dict."
            # assert type(v) is type(d[k]), (
            #     f"Type mismatch for key {k}: {type(d[k])} vs {type(v)}"
            # )
            if type(v) in (jnp.ndarray, np.ndarray):
                assert k.shape == d[k].shape, (
                    f"Shape mismatch for key {k}: {k.shape} vs {d[k].shape}"
                )
                assert k.dtype == d[k].dtype, (
                    f"dtype mismatch for key {k}: {k.dtype} vs {d[k].dtype}"
                )
            d[k] = v
    return d


def tree_stack(trees, axis=0, concatenate=False):
    """Take a list of trees and stack every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function. Taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    _op = jnp.concatenate if concatenate else jnp.stack
    return jax.tree.map(lambda *leaves: _op(leaves, axis=axis), *trees)


class AimLogger(DummyLogger):
    """Wandb-like interface for aim."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "AimLogger"

    @property
    def run_id(self):
        """Return the run hash as ID."""
        return self.run.hash

    @override
    def __init__(
        self, hparams: LoggableConfig, run_name: str | None = None, run_hash=None
    ):
        """Create aim run."""
        global aim
        import aim

        self.run = aim.Run(
            experiment=hparams.project_name,
            repo=hparams.repo,
            run_hash=run_hash,
            log_system_params=True,
        )
        self.run_artifacts_dir = os.path.join("artifacts/aim", self.run.hash)
        self.run.set_artifacts_uri("file:///" + self.run_artifacts_dir)
        hparams = hparams or {}
        if isinstance(hparams, Namespace):
            hparams = vars(hparams)
        elif not isinstance(hparams, dict):
            # Assuming it is a dataclass
            hparams = asdict(hparams)
        self.log_params(hparams)
        if run_name is not None:
            self.run.name = run_name + " " + self.run.hash
        if hparams.get("save_model", False):
            import orbax.checkpoint

            self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    @override
    def log(self, metrics: dict, step=None, context=None):
        """Loop over scalars and track them with aim."""
        for k, v in metrics.items():
            self.run.track(
                np.array(v),
                name=k,
                epoch=None if step is None else int(step),
                context=context,
            )

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        if self.run["hparams"] is None:
            self.run["hparams"] = {}
        self.run["hparams"].update(params_dict)

    def log_dist(self, values: dict, step=None, context=None):
        """Log the given distribution with aim."""
        # TODO: allow sequences.Distributions
        for k, v in values.items():
            self.run.track(
                aim.Distribution(v),
                name=k,
                epoch=None if step is None else int(step),
                context=context,
            )

    def __setitem__(self, key, value):
        """Log scalar for aim."""
        if not isinstance(value, dict):
            # Attempt conversion to float if not a dict
            value = float(value)
        self.run[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        return self.run[key]

    @override
    def finalize(self, ret_code: int = 0, all_param_norms=None, x_vals=None):
        """Finalize the Run."""
        if all_param_norms:
            import plotly.express as px

            all_param_norms = tree_stack(all_param_norms)
            self.log(
                {
                    f"Params/{k}": aim.Figure(
                        px.line(
                            x=x_vals, y=list(v.values()), title=k, labels=list(v.keys())
                        )
                    )
                    for k, v in all_param_norms.items()
                    if v
                }
            )
        if ret_code == 0:
            self.run.report_successful_finish()
        self.run.close()

    @override
    def log_model(self, name, path):
        """Save a file."""
        # FIXME: aim file logging buggy, should be on disc anyway
        # self.run.log_artifact(path, name=name)

    @override
    def log_img(self, name, img, step=None, caption="", pil_mode="RGB", format="png"):
        """Log an image to wandb."""
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, plt.Figure):
            img.canvas.draw()  # Needed on macOS
            img = Image.fromarray(
                np.asarray(img.canvas.buffer_rgba(), dtype=np.uint8), mode="RGBa"
            ).convert(pil_mode)
        self.log(
            {name: aim.Image(img, caption=caption, format=format)},
            step=step,
        )

    def log_figure(self, name, fig, step=None):
        """Log a figure to aim."""
        self.log({name: aim.Figure(fig)}, step=step)

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to aim.

        Parameters
        ----------
        name : str
            Name of the logged object.
        frames : array
            dimension are (frames, height, width, channels)
        step : int, optional
            Step number, by default framework will use global step.
        fps : int, optional
            FPS for the video, by default 30
        caption : str, optional
            Caption for the video, by default
        """
        if not len(frames):
            print("No frames to log for video.")
            return
        file_name = name.replace("/", "_")
        file_name = (
            f"{file_name}_{step}.gif" if step is not None else f"{file_name}.gif"
        )
        file_name = os.path.join(self.run_artifacts_dir, file_name)
        images = [Image.fromarray(frames[i]) for i in range(len(frames))]
        os.makedirs(self.run_artifacts_dir, exist_ok=True)
        images[0].save(
            file_name,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        self.log({name: aim.Image(file_name, caption=caption, format="gif")}, step=step)


WANDB_ENTITY = "franzknut"


class WandbLogger(DummyLogger):
    """Wandb-like interface for aim."""

    @property
    def run_id(self):
        """Return the run hash as ID."""
        return self.run.id

    def __init__(self, hparams: LoggableConfig, run_name: str | None = None):
        """Make WandbLogger.

        Parameters
        ----------
        hparams : LoggableConfig
            Configuration for the run.
        run_name : str | None, optional
            Name for the run in wandb.
        """
        global wandb
        import wandb

        self.run = wandb.init(
            name=run_name,
            project=hparams.project_name,
            config=hparams,
            entity=hparams.repo,
            mode="disabled" if hparams.debug else "online",
            dir="artifacts/log/",
            save_code=False,
        )

        # HACK: Backward compatibility
        if "decay_type" in self.run.config.get("optimizer_config", {}):
            self.run.config["optimizer_config"]["lr_decay_type"] = self.run.config[
                "optimizer_config"
            ]["decay_type"]
            del self.run.config["optimizer_config"]["decay_type"]

        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        self.hparams = hparams.from_dict(
            update_nested_dict(hparams.to_dict(), self.run.config, strict=False),
            drop_extra_fields=False,
        )
        if hparams.log_code:
            self.run.log_code()

    @override
    def log(self, metrics, step=None, context=None):
        """Log metrics to wandb."""
        self.run.log(metrics, step=step)

    def log_dist(self, values: dict, step=None, context=None):
        """Log the given distribution with wandb."""
        # TODO: allow sequences.Distributions
        values = {k: wandb.Histogram(v) for k, v in values.items()}
        self.run.log(values, step=step)

    def __setitem__(self, key, value):
        """Log scalar for wandb."""
        self.run.summary[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        return self.run.summary[key]

    @override
    def flush(self):
        """Flush the logs."""
        wandb.Api().flush()

    @override
    def finalize(self, ret_code: int = 0, all_param_norms: dict = None, x_vals=None):
        """Make lineplots for all items in all_param_norms."""
        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            self.run.log(
                {
                    f"Params/{k}": wandb.plot.line_series(
                        xs=x_vals,
                        ys=v.values(),
                        title=k,
                        keys=list(v.keys()),
                    )
                    for k, v in all_param_norms.items()
                }
            )
        self.run.finish(ret_code)

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        self.run.config.update(params_dict, allow_val_change=True)

    @override
    def log_model(self, name, path, type="model"):
        """Upload a file to wandb."""
        artifact = wandb.Artifact(name.replace("/", "-"), type=type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        elif os.path.isfile(path):
            artifact.add_file(path)
        else:
            print(f"ERROR: Path {path} does not exist, cannot log model.")
            return
        self.run.log_artifact(artifact)
        print(f"WandbLogger: Logged model {name} from {path} as artifact.")

    @override
    def log_img(self, name, img, step=None, caption="", pil_mode="RGB", format=None):
        """Log an image to wandb."""
        self.log(
            {name: wandb.Image(img, caption=caption, mode=pil_mode)},
            step=step,
        )

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb.

        Parameters
        ----------
        name : str
            Name of the logged object.
        frames : array
            dimension are (frames, height, width, channels)
        step : int, optional
            Step number, by default framework will use global step.
        fps : int, optional
            FPS for the video, by default 30
        caption : str, optional
            Caption for the video, by default
        """
        frames = frames.transpose(
            0, 3, 1, 2
        )  # Convert to (frames, channels, height, width)
        self.run.log(
            {name: wandb.Video(frames, fps=fps, caption=caption, format="gif")},
            step=step,
        )


class NamespacedLogger:
    """Log one leg into a shared run under `prefix/`, on its own step axis.

    Metrics, summary keys and params all go under `prefix`, so they don't
    collide with the training leg's. Steps restart at 0 as ``prefix/step``,
    which wandb's charts use as their x-axis, since the run's own step only
    moves forward. Everything else passes through to `logger`.
    """

    def __init__(self, logger, prefix: str):
        self._logger = logger
        self._prefix = prefix
        if isinstance(logger, WandbLogger):
            logger.run.define_metric(f"{prefix}/step")
            logger.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def log(self, metrics: dict, step=None, **kwargs):
        """Log `metrics` under the prefix, with `step` as ``prefix/step``."""
        metrics = {f"{self._prefix}/{k}": v for k, v in metrics.items()}
        if step is not None:
            metrics[f"{self._prefix}/step"] = step
        self._logger.log(metrics, **kwargs)

    def log_params(self, params_dict: dict):
        """Log `params_dict` nested under the prefix."""
        self._logger.log_params({self._prefix: params_dict})

    def __setitem__(self, key, value):
        self._logger[f"{self._prefix}/{key}"] = value

    def __getitem__(self, key):
        return self._logger[f"{self._prefix}/{key}"]


class ExceptionPrinter(contextlib.AbstractContextManager):
    """Hacky way to print exceptions in wandb agent."""

    def __enter__(self):  # noqa
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return False


def make_logger(hparams: LoggableConfig, run_name=""):
    """Make logger according to hparams.

    Parameters
    ----------
    hparams : LoggableConfig
        Hyperparameters for the run. If dict, pick hparams by project_name.
        Will be updated by wandb.config if called by wandb.agent.
    run_name : str, optional
        Name of the run, by default "".

    Returns
    -------
    Any
        Logger instance
    """
    if hparams.logging == "wandb":
        logger: WandbLogger = WandbLogger(hparams, run_name)
    elif hparams.logging == "aim":
        logger = AimLogger(hparams, run_name=run_name)
    else:
        print("No logger specified, using DummyLogger")
        logger = DummyLogger()
    return logger


def with_logger(func: Callable, hparams: LoggableConfig, run_name=""):
    """Wrap training function with logger.

    Parameters
    ----------
    func : Callable
        Function to evaluate.
    hparams : LoggableConfig
    run_name : str, optional

    Returns
    -------
    Any
        Result of the function
    """
    logger = make_logger(hparams, run_name=run_name)
    if hparams.logging == "wandb":
        # Potentially get the replaced hparams for the sweep
        # hparams = logger.hparams
        hparams = hparams.from_dict(dict(logger.run.config))

    # Run the function with the logger
    try:
        ret_code = 0
        with ExceptionPrinter():
            return func(hparams, logger=logger)
    except BaseException as e:
        traceback.print_exception(e)
        ret_code = 1
        raise e
    finally:
        logger.finalize(ret_code)


def get_keystr(k):
    """Even prettier key string."""

    def _str(_k):
        if hasattr(_k, "key"):
            return _k.key
        return str(_k)

    return "/".join(map(_str, k))


def leaf_norms(tree):
    """Return Dict of leaf names and their norms."""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {get_keystr(k): v for k, v in flattened}
    return {
        k: tree_reduce(lambda x, y: x + jnp.linalg.norm(y), v, initializer=0)
        for k, v in flattened.items()
    }


def leaf_means(tree):
    """Return Dict of leaf names and their means."""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {get_keystr(k): v for k, v in flattened}
    return {k: jnp.mean(v) for k, v in flattened.items()}


def tree_norm(tree, **kwargs):
    """Sum of the norm of all elements in the tree."""
    return tree_reduce(
        lambda x, y: x + jnp.linalg.norm(y, **kwargs), tree, initializer=0
    )


def calc_norms(norm_params: dict = {}, leaf_norm_params: dict = {}):
    """Compute norms and leaf norms of given dict of pytrees."""
    norms = {k: tree_norm(v) for k, v in norm_params.items()}
    param_norms = {k: leaf_norms(v) for k, v in leaf_norm_params.items()}
    return norms, param_norms


def log_norms(pytree):
    """Compute norms and leaf norms of given pytree."""
    flattened, _ = jtu.tree_flatten_with_path(pytree)
    flattened = {get_keystr(k): v for k, v in flattened}
    return calc_norms(flattened)


def flatten_params(params):
    """Flatten the given params dictionary."""
    flattened, _ = jtu.tree_flatten_with_path(params)
    return {get_keystr(k): v for k, v in flattened}


def deep_replace(obj, /, **kwargs):
    """Like dataclasses.replace but can replace arbitrarily nested attributes."""
    for k, v in kwargs.items():
        k = k.replace("__", ".")

        while "." in k:
            prefix, _, attr = k.rpartition(".")
            deep_attr = attrgetter(prefix)(obj)
            v = replace(deep_attr, **{attr: v})
            k = prefix
        obj = replace(obj, **{k: v})
    return obj
