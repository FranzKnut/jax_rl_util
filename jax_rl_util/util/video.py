"""Write frame arrays to video or gif. Requires ``opencv-python`` and ``Pillow``."""

import os

import numpy as np


def make_video(frames: np.ndarray, fps: int = 30, output_file: str = "video.mp4"):
    """Write frames of shape (T, H, W, 3) as an mp4.

    Parameters
    ----------
    frames : np.ndarray
        uint8 frames, (T, H, W, 3), BGR.
    fps : int, optional
        Frames per second, by default 30.
    output_file : str, optional
        Output path, by default "video.mp4".
    """
    import cv2

    shape = frames.shape[2], frames.shape[1]
    print("Writing video to", output_file)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, shape)
    for f in frames:
        out.write(f)
    out.release()


def make_gif(
    frames: np.ndarray,
    fps: int = 30,
    output_file: str = "artifacts/videos/video.gif",
    scale: float = 0.5,
    downsample: int = 5,
):
    """Write frames of shape (T, H, W, 3) as a gif.

    Parameters
    ----------
    frames : np.ndarray
        uint8 frames, (T, H, W, 3), RGB.
    fps : int, optional
        Frames per second of the source, by default 30.
    output_file : str, optional
        Output path, by default "artifacts/videos/video.gif".
    scale : float, optional
        Resize factor, by default 0.5.
    downsample : int, optional
        Keep every n-th frame, by default 5.
    """
    import cv2
    from PIL import Image

    new_shape = [int(s * scale) for s in reversed(frames.shape[1:-1])]
    imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).resize(new_shape) for img in frames]
    print("Writing gif to", output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    imgs[0].save(
        output_file, save_all=True, append_images=imgs[1::downsample], duration=1000 // (fps * downsample), loop=0
    )
