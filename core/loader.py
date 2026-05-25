"""
core/loader.py
--------------
Image loading and Ruifrok-Johnston colour deconvolution.
No GUI imports.
"""
from dataclasses import dataclass
import numpy as np
from skimage import io
from skimage.color import rgb2hed


@dataclass
class ImageData:
    """Holds the original and deconvolved image channels."""
    img: np.ndarray      # float32 RGB [0, 1]
    H: np.ndarray        # hematoxylin channel
    E: np.ndarray        # eosin channel
    raw_dtype: str       # original dtype string
    raw_max: float       # original max value (diagnostic)


def load_image(path: str) -> ImageData:
    """
    Load any common histology format into float32 RGB [0, 1].
    Handles uint8, uint16, float images and RGBA.

    Returns an ImageData with H and E channels separated.
    Raises ValueError if the file cannot be read.
    """
    raw = io.imread(path)

    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    elif raw.ndim == 3 and raw.shape[2] > 3:
        raw = raw[:, :, :3]

    raw_dtype = str(raw.dtype)
    raw_max = float(raw.max())

    # Dtype-aware normalisation to [0, 1]
    dtype = raw.dtype
    if dtype == np.uint8:
        img = raw.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        img = raw.astype(np.float32) / 65535.0
    elif dtype == np.uint32:
        img = raw.astype(np.float32) / 4294967295.0
    elif np.issubdtype(dtype, np.floating):
        img = raw.astype(np.float32)
        if img.max() > 1.0:
            img /= (255.0 if img.max() <= 255.0 else img.max())
    else:
        vmin, vmax = float(raw.min()), float(raw.max())
        img = (raw.astype(np.float32) - vmin) / (vmax - vmin + 1e-8)

    img = np.clip(img, 0.0, 1.0)

    hed = rgb2hed(img)
    H = hed[:, :, 0].copy()
    E = hed[:, :, 1].copy()

    return ImageData(img=img, H=H, E=E, raw_dtype=raw_dtype, raw_max=raw_max)
