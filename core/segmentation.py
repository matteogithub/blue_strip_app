"""
core/segmentation.py
--------------------
Tissue detection, lobe labelling, and distance-from-capsule computation.
No GUI imports.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, remove_small_objects, disk

from config import Config
from core.loader import ImageData


@dataclass
class SegmentationResult:
    """Output of the tissue segmentation stage."""
    tissue_mask: np.ndarray       # bool (H×W)
    lobe_labels: np.ndarray       # int  (H×W), 0 = background
    lobes: list                   # list of skimage regionprops, largest first
    dist_from_bg: np.ndarray      # float (H×W), depth below outer capsule (px)


def segment(image_data: ImageData, config: Config,
            roi_mask: Optional[np.ndarray] = None) -> SegmentationResult:
    """
    Detect tissue, label connected lobes, and compute the
    distance-from-capsule map.

    Steps:
      1. Invert mean-RGB → Otsu threshold → binary tissue mask
      2. Morphological cleanup (closing, small-object removal)
      3. Label connected components, filter by min area
      4. Absorb interior fragments into enclosing lobes
      5. Fill internal cavities (renal sinus) per lobe before EDT
    """
    img = image_data.img
    total_px = img.shape[0] * img.shape[1]
    close_px = max(1, int(config.close_um / config.pixel_size_um))
    min_px = int(total_px * config.min_lobe_frac)

    # ── 1. Otsu threshold on inverted image ──────────────────────────────────
    inverted = 1.0 - np.mean(img, axis=-1)
    thresh = threshold_otsu(inverted)
    mask = inverted > thresh
    if roi_mask is not None:
        if roi_mask.shape != img.shape[:2]:
            raise ValueError(
                f"roi_mask shape {roi_mask.shape} does not match image "
                f"shape {img.shape[:2]}. Reload the image and redraw the ROI."
            )
        mask = mask & roi_mask

    # ── 2. Morphological cleanup ─────────────────────────────────────────────
    mask = binary_closing(mask, disk(close_px))
    mask = remove_small_objects(mask, min_size=max(50, int(total_px * 0.0001)))

    # ── 3. Label and size-filter primary lobes ────────────────────────────────
    raw_labels = label(mask)
    all_props = sorted(regionprops(raw_labels), key=lambda p: p.area, reverse=True)

    valid = [p.label for p in all_props if p.area >= min_px]
    if not valid:
        valid = [p.label for p in all_props[:3]]  # fallback: keep 3 largest

    primary = np.isin(raw_labels, valid)
    lobe_labels = label(primary)

    # ── 4. Absorb interior orphan fragments ──────────────────────────────────
    lobes = sorted(regionprops(lobe_labels), key=lambda p: p.area, reverse=True)
    for p in lobes:
        lmask = lobe_labels == p.label
        filled = binary_fill_holes(lmask)
        interior = filled & ~lmask & (lobe_labels == 0)
        lobe_labels[interior] = p.label

    lobes = sorted(regionprops(lobe_labels), key=lambda p: p.area, reverse=True)
    tissue_mask = lobe_labels > 0

    # ── 5. Distance transform (internal cavities filled) ─────────────────────
    # binary_fill_holes runs on the full per-lobe mask so the renal sinus
    # is always correctly filled.  The ROI is applied after, so the EDT
    # treats pixels outside the ROI as background.
    tissue_filled = np.zeros_like(tissue_mask, dtype=bool)
    for p in lobes:
        tissue_filled |= binary_fill_holes(lobe_labels == p.label)
    if roi_mask is not None:
        tissue_filled = tissue_filled & roi_mask
    dist_from_bg = distance_transform_edt(tissue_filled)

    return SegmentationResult(
        tissue_mask=tissue_mask,
        lobe_labels=lobe_labels,
        lobes=lobes,
        dist_from_bg=dist_from_bg,
    )

