"""
core/strategies/blobs.py
------------------------
Strategy C: LoG blob detection on the eosin channel.
Boundary = valley between near-capsule non-glomerular cluster
           and the main glomerular depth distribution.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.feature import blob_log

from core.strategies.base import BaseStrategy, StrategyResult


class StrategyBlobs(BaseStrategy):
    """Strategy C — LoG blob distribution."""

    @property
    def label(self) -> str:
        return "C"

    @property
    def name(self) -> str:
        return "LoG Blob Distribution"

    def run(self, image_data, lobe_mask, dist_from_bg, config) -> StrategyResult:
        px = config.pixel_size_um
        E = image_data.E

        # Normalise E within tissue
        tissue_vals = E[lobe_mask]
        if tissue_vals.size == 0:
            return self._empty_result()
        e_lo, e_hi = np.percentile(tissue_vals, [1, 99])
        E_norm = np.clip((E - e_lo) / (e_hi - e_lo + 1e-8), 0, 1)
        E_norm[~lobe_mask] = 0.0

        min_sigma = config.sc_min_r_um / (px * np.sqrt(2))
        max_sigma = config.sc_max_r_um / (px * np.sqrt(2))

        blobs = blob_log(
            E_norm,
            min_sigma=min_sigma, max_sigma=max_sigma,
            num_sigma=config.sc_n_sigma,
            threshold=config.sc_blob_threshold,
            overlap=config.sc_max_overlap,
        )

        # Filter to this lobe and above minimum depth
        H_img, W_img = lobe_mask.shape
        valid_blobs = []
        for cy, cx, sigma in blobs:
            cy, cx = int(round(cy)), int(round(cx))
            if not (0 <= cy < H_img and 0 <= cx < W_img):
                continue
            if not lobe_mask[cy, cx]:
                continue
            depth_um = float(dist_from_bg[cy, cx]) * px
            if depth_um < config.sc_min_depth_um:
                continue
            valid_blobs.append({
                "cy": cy, "cx": cx,
                "r_px": sigma * np.sqrt(2),
                "depth_um": depth_um,
            })

        if not valid_blobs:
            return self._empty_result()

        depths = sorted(b["depth_um"] for b in valid_blobs)
        boundary_um, valley_info = self._find_valley(
            depths, fallback_percentile=config.sc_depth_percentile)

        # Build pseudo-profile for display (histogram as profile)
        bin_w = 15.0
        bins = np.arange(0, max(depths) + bin_w, bin_w)
        counts, edges = np.histogram(depths, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2.0

        return StrategyResult(
            label=self.label, name=self.name,
            boundary_um=boundary_um,
            centres=centres,
            profile=counts.astype(float),
            smoothed=gaussian_filter1d(counts.astype(float),
                                       sigma=config.profile_smooth_sigma),
            extra={"blobs": valid_blobs, "valley_info": valley_info,
                   "depths": depths,
                   "p15_um": float(np.percentile(depths, config.sc_depth_percentile))},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_result() -> StrategyResult:
        return StrategyResult(
            label="C", name="LoG Blob Distribution",
            boundary_um=float("nan"),
            centres=np.array([]), profile=np.array([]), smoothed=np.array([]),
            extra={"blobs": [], "depths": []},
        )

    @staticmethod
    def _find_valley(depths, bin_width=15.0, smooth_sigma=2.0,
                     min_peak_count=5, fallback_percentile=15):
        """Valley between the near-capsule cluster and the glomerular peak."""
        d = np.array(depths)
        bins = np.arange(0, d.max() + bin_width, bin_width)
        counts, edges = np.histogram(d, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2.0
        smoothed = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)

        peaks, _ = find_peaks(smoothed, height=min_peak_count, distance=2)
        if len(peaks) == 0:
            return (float(np.percentile(d, fallback_percentile)),
                    {"method": f"P{fallback_percentile}_fallback"})

        valleys, _ = find_peaks(-smoothed[peaks[0]:], distance=2)
        if len(valleys) == 0:
            return (float(np.percentile(d, fallback_percentile)),
                    {"method": f"P{fallback_percentile}_fallback"})

        v_idx = valleys[0] + peaks[0]
        return float(centres[v_idx]), {
            "method": "valley",
            "first_peak_um": float(centres[peaks[0]]),
            "valley_um": float(centres[v_idx]),
        }
