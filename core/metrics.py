"""
core/metrics.py
---------------
Compute and export per-lobe quantitative metrics.
"""
import csv
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np

from core.strategies.base import StrategyResult
from core.fusion import FusionResult
from config import Config


@dataclass
class LobeMetrics:
    image_path: str
    lobe_label: int
    timestamp: str
    pixel_size_um: float
    # Strategy depths
    depth_A_um: Optional[float]
    depth_B_um: Optional[float]
    depth_C_um: Optional[float]
    # Fusion
    depth_fused_um: float
    strategy_agreement_um: float
    # Spatial
    spatial_mean_um: float
    spatial_std_um: float
    cv_pct: float
    min_thickness_um: float
    max_thickness_um: float
    # Area
    strip_area_mm2: float
    strip_frac_pct: float
    # Glomerular metrics (None when Strategy C found no blobs)
    gl_n_raw:             Optional[int]   = None
    gl_n_filtered:        Optional[int]   = None
    gl_density_per_mm2:   Optional[float] = None
    gl_cortical_area_mm2: Optional[float] = None
    gl_argc_mean:         Optional[float] = None
    gl_argc_sd:           Optional[float] = None
    gl_generation_count:  Optional[int]   = None


def compute_metrics(image_path: str,
                    lobe_label: int,
                    lobe_mask: np.ndarray,
                    dist_from_bg: np.ndarray,
                    strategy_results: List[StrategyResult],
                    fusion: FusionResult,
                    config: Config,
                    glom_result=None) -> LobeMetrics:
    """Build a LobeMetrics from all analysis outputs."""
    px = config.pixel_size_um

    depths = {r.label: r.boundary_um for r in strategy_results}
    agreement = float(np.ptp(fusion.estimates)) if len(fusion.estimates) > 1 else 0.0

    thick = fusion.thickness_um
    if len(thick):
        mean_t = float(np.mean(thick))
        std_t  = float(np.std(thick))
        cv     = 100 * std_t / mean_t if mean_t > 0 else float("nan")
        min_t  = float(np.min(thick))
        max_t  = float(np.max(thick))
    else:
        mean_t = std_t = cv = min_t = max_t = float("nan")

    bnd_px = fusion.boundary_um / px if not np.isnan(fusion.boundary_um) else 0
    strip_px  = (lobe_mask & (dist_from_bg < bnd_px)).sum()
    lobe_px   = lobe_mask.sum()
    strip_mm2 = strip_px * px**2 / 1e6
    lobe_mm2  = lobe_px  * px**2 / 1e6
    frac      = 100 * strip_mm2 / lobe_mm2 if lobe_mm2 > 0 else float("nan")

    metrics = LobeMetrics(
        image_path=image_path,
        lobe_label=lobe_label,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        pixel_size_um=px,
        depth_A_um=depths.get("A"),
        depth_B_um=depths.get("B"),
        depth_C_um=depths.get("C"),
        depth_fused_um=fusion.boundary_um,
        strategy_agreement_um=agreement,
        spatial_mean_um=mean_t,
        spatial_std_um=std_t,
        cv_pct=cv,
        min_thickness_um=min_t,
        max_thickness_um=max_t,
        strip_area_mm2=strip_mm2,
        strip_frac_pct=frac,
    )
    if glom_result is not None:
        metrics.gl_n_raw             = glom_result.n_raw
        metrics.gl_n_filtered        = glom_result.n_filtered
        metrics.gl_density_per_mm2   = glom_result.density["density_per_mm2"]
        metrics.gl_cortical_area_mm2 = glom_result.density["cortical_area_mm2"]
        metrics.gl_argc_mean         = glom_result.argc["argc_mean"]
        metrics.gl_argc_sd           = glom_result.argc["argc_sd"]
        metrics.gl_generation_count  = glom_result.depth_profile["generation_count"]
    return metrics


def export_csv(metrics_list: List[LobeMetrics], path: str) -> None:
    """Append metrics rows to a CSV file (creates header if file is new)."""
    if not metrics_list:
        return
    fieldnames = list(asdict(metrics_list[0]).keys())
    import os
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for m in metrics_list:
            writer.writerow(asdict(m))
