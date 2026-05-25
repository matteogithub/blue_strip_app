"""
core/strategies/density.py
--------------------------
Strategy B: hematoxylin-positive pixel area fraction per depth shell.
No nucleus segmentation needed — robust at any magnification.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu

from core.strategies.base import BaseStrategy, StrategyResult


class StrategyDensity(BaseStrategy):
    """Strategy B — Nuclear area fraction."""

    @property
    def label(self) -> str:
        return "B"

    @property
    def name(self) -> str:
        return "Nuclear Area Fraction"

    def run(self, image_data, lobe_mask, dist_from_bg, config) -> StrategyResult:
        H = image_data.H
        tissue_pixels = H[lobe_mask]
        thresh = threshold_otsu(tissue_pixels) if tissue_pixels.size > 0 else 0.03
        binary_H = (H > thresh) & lobe_mask

        edges, centres = self._shell_edges(config)

        fractions = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            shell = lobe_mask & (dist_from_bg >= lo) & (dist_from_bg < hi)
            n_total = shell.sum()
            if n_total < 10:
                fractions.append(np.nan)
            else:
                fractions.append(float(binary_H[shell].sum()) / n_total)

        fractions = np.array(fractions, dtype=float)
        valid = ~np.isnan(fractions)
        smoothed = np.full_like(fractions, np.nan)
        if valid.any():
            smoothed[valid] = gaussian_filter1d(
                fractions[valid], sigma=config.profile_smooth_sigma)

        boundary_um = self._find_boundary(centres, smoothed)
        return StrategyResult(
            label=self.label, name=self.name,
            boundary_um=boundary_um,
            centres=centres, profile=fractions, smoothed=smoothed,
            extra={"h_threshold": thresh},
        )
