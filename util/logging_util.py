"""Utilies for logging."""

import collections
import contextlib
import os
import traceback
from argparse import Namespace
from dataclasses import asdict, dataclass, replace
from operator import attrgetter
from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import simple_parsing
from matplotlib import pyplot as plt
from PIL import Image
from typing_extensions import override


@dataclass
class LoggableConfig(simple_parsing.Serializable, decode_into_subclasses=True):
    """Base configuration for experiments logged to wandb or aim."""

    logging: str | None = "aim"
    repo: str | None = None  # Used for entity in wandb
    project_name: str | None = "default"
    debug: bool | int = False
    log_code: bool = False


class DummyLogger(dict, object):
    """Dummy Logger that does nothing besides acting as dictionary."""

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

    def log_params(self, params_dict):
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

    def finalize(self, all_param_norms=None):
        """Log additional plots or media.

        Parameters
        ----------
        all_param_norms : TODO
            _description_
        """
        pass

    def log_model(self, name: str, path: str):
        """Save a file as an artifact.

        Parameters
        ----------
        name : str
            Name of the artifact.
        path : str
            Path to the file to be logged.
        """
        pass

    def log_img(self, name, img, step=None, caption="", pil_mode="RGB"):
        """Log an image to wandb."""

    def log_video(self, name: str, frames, step: int = None, fps=4, **kwargs):
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
        pass


def update_nested_dict(d, u):
    """Update nested dict d with values from nested dict u.

    Parameters
    ----------
    d : dict
        Base dict
    u : dict
        Updates

    Returns
    -------
    dict
        d with values overwritten by u
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def tree_stack(trees):
    """Take a list of trees and stack every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function. Taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


class AimLogger(DummyLogger):
    """Wandb-like interface for aim."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "AimLogger"

    @override
    def __init__(self, name, repo=None, hparams=None, run_name="", run_hash=None):
        """Create aim run."""
        global aim
        import aim

        self.run = aim.Run(experiment=name, repo=repo, run_hash=run_hash, log_system_params=True)
        self.run_artifacts_dir = os.path.join("artifacts/aim", self.run.hash)
        self.run.set_artifacts_uri("file:///" + self.run_artifacts_dir)
        hparams = hparams or {}
        if isinstance(hparams, Namespace):
            hparams = vars(hparams)
        elif not isinstance(hparams, dict):
            # Assuming it is a dataclass
            hparams = asdict(hparams)
        self.log_params(hparams)
        self.run.name = run_name + " " + self.run.hash
        if hparams.get("save_model", False):
            import orbax.checkpoint

            self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    @override
    def log(self, metrics: dict, step=None, context=None):
        """Loop over scalars and track them with aim."""
        for k, v in metrics.items():
            self.run.track(v, name=k, epoch=None if step is None else int(step), context=context)

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        self.run["hparams"] = params_dict

    def log_dist(self, values: dict, step=None, context=None):
        """Log the given distribution with aim."""
        # TODO: allow sequences.Distributions
        for k, v in values.items():
            self.run.track(aim.Distribution(v), name=k, epoch=None if step is None else int(step), context=context)

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
    def report_successful_finish(self, all_param_norms=None, x_vals=None):
        """Make lineplots for param norms and block until all metrics are logged."""
        if all_param_norms:
            import plotly.express as px

            all_param_norms = tree_stack(all_param_norms)
            self.log(
                {
                    f"Params/{k}": aim.Figure(px.line(x=x_vals, y=list(v.values()), title=k, labels=list(v.keys())))
                    for k, v in all_param_norms.items()
                    if v
                }
            )

        self.run.report_successful_finish(block=True)

    @override
    def finalize(self, _=None):
        """Finalize the Run."""
        self.run.finalize()

    @override
    def log_model(self, name, path):
        """Save a file."""
        # FIXME: aim file logging buggy, should be on disc anyway
        # self.run.log_artifact(path, name=name)

    @override
    def log_img(self, name, img, step=None, caption="", pil_mode="RGB", format="png"):
        """Log an image to wandb."""
        if isinstance(img, plt.Figure):
            _img = fig_to_array(img)
            pil_mode = "RGBA"
        else:
            _img = np.array(img, dtype=np.uint8)
        self.log(
            {name: aim.Image(Image.fromarray(_img, mode=pil_mode), caption=caption, format=format)},
            step=step,
        )

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        file_name = name.replace("/", "_")
        file_name = f"{file_name}_{step}.gif" if step is not None else f"{file_name}.gif"
        file_name = os.path.join(self.run_artifacts_dir, file_name)
        images = [Image.fromarray(frames[i]) for i in range(len(frames))]
        os.makedirs(self.run_artifacts_dir, exist_ok=True)
        images[0].save(file_name, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
        self.log({name: aim.Image(file_name, caption=caption, format="gif")}, step=step)


class WandbLogger(DummyLogger):
    """Wandb-like interface for aim."""

    @override
    def log(self, metrics, step=None, context=None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)

    def log_dist(self, values: dict, step=None, context=None):
        """Log the given distribution with wandb."""
        # TODO: allow sequences.Distributions
        values = {k: wandb.Histogram(v) for k, v in values.items()}
        wandb.log(values, step=step)

    def __setitem__(self, key, value):
        """Log scalar for wandb."""
        wandb.run.summary[key] = value
        self.flush()

    def __getitem__(self, key):
        """Get value from aim run."""
        self.flush()
        return wandb.run.summary[key]

    @override
    def flush(self):
        """Flush the logs."""
        wandb.Api().flush()

    @override
    def finalize(self, all_param_norms: dict = None, x_vals=None):
        """Make lineplots for all items in all_param_norms."""
        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            wandb.log(
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

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        wandb.run.config.update(params_dict, allow_val_change=True)

    @override
    def log_model(self, name, path):
        """Upload a file to wandb."""
        artifact = wandb.Artifact(name.replace("/", "-"), "model")
        artifact.add_dir(path)
        wandb.log_artifact(artifact)

    @override
    def log_img(self, name, img, step=None, caption="", pil_mode="RGB", format=None):
        """Log an image to wandb."""
        self.log(
            {name: wandb.Image(img, caption=caption, mode=pil_mode)},
            step=step,
        )

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        wandb.log({name: wandb.Video(frames, fps=fps, caption=caption)}, step=step)


class ExceptionPrinter(contextlib.AbstractContextManager):
    """Ensures that exceptions are logged from wandb agents."""

    def __enter__(self):  # noqa
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return False


def wandb_wrapper(project_name, func: Callable | dict[str, Callable], hparams: LoggableConfig):
    """Init wandb and evaluate function.

    Parameters
    ----------
    project_name : str
        Name of the project
    func : Callable | dict[str, Callable]
        Function to evaluate, if dict, pick function by project_name.
    hparams : LoggableConfig
        Hyperparameters for the run.
        Will be updated by wandb.config if called by wandb.agent.
    """
    global wandb
    import wandb

    logger = WandbLogger()

    with (
        wandb.init(
            project=project_name,
            config=hparams,
            mode="disabled" if hparams.debug else "online",
            dir="logs/",
            save_code=False,
            entity=hparams.repo,
        ),
        ExceptionPrinter(),
    ):
        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        hparams = hparams.from_dict(
            update_nested_dict(hparams.to_dict(), wandb.config),
            drop_extra_fields=False,
        )
        if hparams.log_code:
            wandb.run.log_code()

        if isinstance(func, dict):
            func = func[project_name]
        return func(hparams, logger=logger)


def with_logger(
    func: Callable,
    hparams: LoggableConfig,
    run_name="",
):
    """Wrap training function with logger.

    Parameters
    ----------
    func : Callable
        Function to evaluate.
    hparams : LoggableConfig
        Hyperparameters for the run. If dict, pick hparams by project_name.
        Will be updated by wandb.config if called by wandb.agent.
    run_name : str, optional
        Name of the run, by default "".

    Returns
    -------
    Any
        Result of the function
    """
    if hparams.logging == "wandb":

        def pick_fun_and_run(_hparams, logger):
            return func(_hparams, logger=logger)

        # Getting the function is done inside after wandb.init to get sweep config first.
        return wandb_wrapper(hparams.project_name, pick_fun_and_run, hparams)

    elif hparams.logging == "aim":
        # Get the function if it is a dict
        if isinstance(func, dict):
            func = func[hparams.project_name]
        logger = AimLogger(hparams.project_name, repo=hparams.repo, hparams=hparams, run_name=run_name)
        try:
            return func(hparams, logger=logger)
        finally:
            logger.finalize()
    else:
        return func(hparams)


def leaf_norms(tree):
    """Return Dict of leaf names and their norms."""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return {k: jax.tree.reduce(lambda x, y: x + jnp.linalg.norm(y), v, initializer=0) for k, v in flattened.items()}


def tree_norm(tree, **kwargs):
    """Sum of the norm of all elements in the tree."""
    return jax.tree.reduce(lambda x, y: x + jnp.linalg.norm(y, **kwargs), tree, initializer=0)


def calc_norms(norm_params: dict = {}, leaf_norm_params: dict = {}):
    """Compute norms and leaf norms of given dict of pytrees."""
    norms = {k: tree_norm(v) for k, v in norm_params.items()}
    param_norms = {k: leaf_norms(v) for k, v in leaf_norm_params.items()}
    return norms, param_norms


def log_norms(pytree):
    """Compute norms and leaf norms of given pytree."""
    flattened, _ = jtu.tree_flatten_with_path(pytree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return calc_norms(flattened)


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


def fig_to_array(fig: plt.Figure):
    """Convert matplotlib figure to numpy array."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)


# .reshape(
#         fig.canvas.get_width_height()[::-1] + (4,)
#     )
