"""
core/strategies/intensity.py
-----------------------------
Strategy A: mean hematoxylin intensity per depth shell.
Blue strip = sustained high-H plateau; boundary = first significant drop.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from core.strategies.base import BaseStrategy, StrategyResult


class StrategyIntensity(BaseStrategy):
    """Strategy A — Hematoxylin intensity profile."""

    @property
    def label(self) -> str:
        return "A"

    @property
    def name(self) -> str:
        return "H-Intensity Profile"

    def run(self, image_data, lobe_mask, dist_from_bg, config) -> StrategyResult:
        H = image_data.H
        edges, centres = self._shell_edges(config)

        profile = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            shell = lobe_mask & (dist_from_bg >= lo) & (dist_from_bg < hi)
            profile.append(H[shell].mean() if shell.any() else np.nan)

        profile = np.array(profile, dtype=float)
        valid = ~np.isnan(profile)
        smoothed = np.full_like(profile, np.nan)
        if valid.any():
            smoothed[valid] = gaussian_filter1d(
                profile[valid], sigma=config.profile_smooth_sigma)

        boundary_um = self._find_boundary(centres, smoothed)
        return StrategyResult(
            label=self.label, name=self.name,
            boundary_um=boundary_um,
            centres=centres, profile=profile, smoothed=smoothed,
        )
