"""
core/strategies/base.py
-----------------------
Shared result dataclass and abstract base for all three strategies.
Adding Strategy D = create one new file + register it in runner.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class StrategyResult:
    """Standardised output from any strategy."""
    label: str               # e.g. "A", "B", "C"
    name: str                # human-readable name
    boundary_um: float       # estimated blue-strip depth
    centres: np.ndarray      # depth of each shell/point (µm)
    profile: np.ndarray      # raw signal values
    smoothed: np.ndarray     # smoothed signal values
    extra: dict = None       # strategy-specific extras (blobs, valley info, …)

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseStrategy(ABC):
    """
    Common interface for all blue-strip depth strategies.

    Subclasses implement run() which takes the image data,
    the per-lobe binary mask, the distance map, and the config,
    and returns a StrategyResult.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Single-letter identifier, e.g. 'A'."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in the GUI."""

    @abstractmethod
    def run(self, image_data, lobe_mask: np.ndarray,
            dist_from_bg: np.ndarray, config) -> StrategyResult:
        """
        Compute the blue-strip depth estimate for one lobe.

        Parameters
        ----------
        image_data : ImageData
        lobe_mask  : bool (H×W) — True for pixels in this lobe
        dist_from_bg : float (H×W) — depth below capsule in pixels
        config     : Config

        Returns
        -------
        StrategyResult
        """

    # ── Shared helper: robust boundary detection ─────────────────────────────
    @staticmethod
    def _find_boundary(centres: np.ndarray, smoothed: np.ndarray,
                       n_search_frac: float = 0.67) -> float:
        """
        Find the first significant negative gradient (inflection/drop)
        in a smoothed depth profile.

        Guarded against all-NaN search windows (e.g. after region exclusion).
        """
        from scipy.signal import find_peaks
        grad = np.gradient(smoothed)
        n = int(len(grad) * n_search_frac)
        g_sub = grad[:n]                          # search window
        valid = ~np.isnan(g_sub)

        if not valid.any():
            return float(centres[min(2, len(centres) - 1)])

        g_safe = np.where(valid, g_sub, np.inf)
        peaks, _ = find_peaks(-g_safe,
                               height=np.nanstd(g_sub[valid]) * 0.4,
                               distance=2)
        if len(peaks):
            return float(centres[peaks[0]])
        return float(centres[int(np.nanargmin(np.where(valid, g_sub, np.nan)))])

    # ── Shared helper: build depth shells ────────────────────────────────────
    @staticmethod
    def _shell_edges(config) -> tuple:
        """Return (edges_px, centres_um) for depth shells."""
        px = config.pixel_size_um
        shell_px = config.shell_thickness_um / px
        max_px = config.max_depth_um / px
        edges = np.arange(0, max_px + shell_px, shell_px)
        centres = (edges[:-1] + edges[1:]) / 2.0 * px
        return edges, centres
