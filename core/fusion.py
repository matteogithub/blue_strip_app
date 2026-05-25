"""
core/fusion.py
--------------
Median fusion of three strategy estimates and spatial profile d(arc).
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image

from core.strategies.base import StrategyResult
from config import Config


@dataclass
class FusionResult:
    """Output of the fusion stage for one lobe."""
    boundary_um: float              # fused (median) depth estimate
    estimates: List[float]          # individual strategy estimates
    strategy_labels: List[str]      # corresponding labels ("A", "B", "C")
    arc_um: np.ndarray              # arc-length along capsule (µm)
    thickness_um: np.ndarray        # local strip thickness at each contour point


def fuse(strategy_results: List[StrategyResult],
         lobe_mask: np.ndarray,
         dist_from_bg: np.ndarray,
         config: Config) -> FusionResult:
    """
    Combine strategy estimates and compute the spatial profile.

    Parameters
    ----------
    strategy_results : list of StrategyResult (A, B, C)
    lobe_mask        : bool (H×W)
    dist_from_bg     : float (H×W)
    config           : Config
    """
    estimates = [r.boundary_um for r in strategy_results
                 if not np.isnan(r.boundary_um)]
    labels = [r.label for r in strategy_results
              if not np.isnan(r.boundary_um)]

    if not estimates:
        return FusionResult(
            boundary_um=float("nan"),
            estimates=[], strategy_labels=[],
            arc_um=np.array([]), thickness_um=np.array([]),
        )

    boundary_um = float(np.median(estimates))
    arc_um, thickness_um = _spatial_profile(lobe_mask, dist_from_bg,
                                             boundary_um, config)
    return FusionResult(
        boundary_um=boundary_um,
        estimates=estimates,
        strategy_labels=labels,
        arc_um=arc_um,
        thickness_um=thickness_um,
    )


def _spatial_profile(lobe_mask, dist_from_bg, boundary_um, config,
                     n_pts=200):
    """
    Sample strip thickness along the convex-hull contour of the lobe.
    Uses inward-normal ray casting from each contour point.
    """
    px = config.pixel_size_um
    try:
        hull = convex_hull_image(lobe_mask)
    except Exception:
        hull = lobe_mask

    contours = find_contours(hull.astype(float), level=0.5)
    if not contours:
        return np.array([]), np.array([])
    contour = max(contours, key=len)

    idx = np.round(np.linspace(0, len(contour) - 1, n_pts)).astype(int)
    pts = contour[idx]

    bnd_px = boundary_um / px
    H_img, W_img = lobe_mask.shape
    max_steps = int(bnd_px) + 10
    thickness = []

    for i_pt in range(len(pts)):
        r, c = pts[i_pt]
        prev = pts[i_pt - 1]
        nxt = pts[(i_pt + 1) % len(pts)]
        tangent = nxt - prev
        tn = np.linalg.norm(tangent) + 1e-8
        tangent /= tn
        na = np.array([-tangent[1],  tangent[0]])
        nb = np.array([ tangent[1], -tangent[0]])

        da = db = 0
        for st in [3, 6, 10]:
            ra, ca = int(round(r + st*na[0])), int(round(c + st*na[1]))
            rb, cb = int(round(r + st*nb[0])), int(round(c + st*nb[1]))
            da = dist_from_bg[ra, ca] if 0<=ra<H_img and 0<=ca<W_img else 0
            db = dist_from_bg[rb, cb] if 0<=rb<H_img and 0<=cb<W_img else 0
            if da != db:
                break
        inward = na if da > db else nb

        local_thick = 0.0
        for step in range(1, max_steps + 1):
            ri = int(round(r + step * inward[0]))
            ci = int(round(c + step * inward[1]))
            if not (0 <= ri < H_img and 0 <= ci < W_img):
                local_thick = (step - 1) * px
                break
            if not lobe_mask[ri, ci]:
                local_thick = (step - 1) * px
                break
            if dist_from_bg[ri, ci] >= bnd_px:
                local_thick = step * px
                break
        else:
            local_thick = max_steps * px

        thickness.append(min(local_thick, boundary_um))

    thickness = np.array(thickness)
    diffs = np.diff(pts, axis=0) * px
    arc = np.concatenate([[0], np.sqrt((diffs**2).sum(axis=1)).cumsum()])
    return arc, thickness
